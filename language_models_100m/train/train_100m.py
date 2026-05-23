import argparse
import json
import logging
import math
import os
import shutil
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin, set_seed as accelerate_set_seed
from datasets import Dataset
from module import build_krause_attention
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from tqdm import tqdm
from training_utils import (
    RevisionLMDataCollator,
    compute_revision_lm_loss,
    count_parameters,
    data_loader_kwargs,
    eval_revision_loss_accelerate,
    find_latest_revision_checkpoint,
    init_distributed,
    list_revision_checkpoint_steps,
    load_or_tokenize_packed,
    load_text_splits,
    revision_model_subdir,
)
from transformers import AutoTokenizer
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
from torch.utils.checkpoint import checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
DEFAULT_DATA_DIR = "./datasets/fwe10bt"

class TransformerBlock(nn.Module):
    def __init__(self, cfg: "ModelConfig"):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size)
        self.ln2 = nn.LayerNorm(cfg.hidden_size)
        self.attn = build_krause_attention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            attn_dropout=cfg.attn_dropout,
            window_size=cfg.window_size,
            top_k=cfg.top_k,
            init_sigma=cfg.init_sigma,
            init_standard_weight=cfg.init_standard_weight,
        )

        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.ffn_hidden_size),
            nn.GELU(),
            nn.Linear(cfg.ffn_hidden_size, cfg.hidden_size),
            nn.Dropout(cfg.resid_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int = 768
    num_layers: int = 8
    num_heads: int = 12
    ffn_hidden_size: int = 2048
    max_seq_len: int = 1024
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    emb_dropout: float = 0.0

    window_size: int = 256
    top_k: int = 192
    init_sigma: float = 4.0
    init_standard_weight: float = 0.8
    gradient_checkpointing: bool = True


class CausalTransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.hidden_size)
        self.emb_dropout = nn.Dropout(cfg.emb_dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_ln = nn.LayerNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.cfg.max_seq_len}")

        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        x = self.emb_dropout(x)
        use_ckpt = self.cfg.gradient_checkpointing and self.training
        for blk in self.blocks:
            if use_ckpt:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        x = self.final_ln(x)
        logits = self.lm_head(x)
        return logits

def build_fsdp_plugin_for_revision() -> FullyShardedDataParallelPlugin:
    auto_wrap = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )
    return FullyShardedDataParallelPlugin(
        auto_wrap_policy=auto_wrap,
        activation_checkpointing=True,
        sync_module_states=True,
        use_orig_params=True,
    )

def save_revision_rotating_checkpoint(
    accelerator: Accelerator,
    model: nn.Module,
    tokenizer,
    cfg: ModelConfig,
    out_dir: Path,
    optimizer_step: int,
    checkpoint_history: List[int],
    save_total_limit: int,
    model_tag: str,
) -> None:
    accelerator.wait_for_everyone()
    ckpt_dir = out_dir / f"checkpoint-{optimizer_step}-{model_tag}"
    accelerator.save_state(str(ckpt_dir))
    accelerator.wait_for_everyone()
    hf_dir = ckpt_dir / f"hf_model-{model_tag}"
    accelerator.save_model(model, hf_dir)
    if accelerator.is_main_process:
        with open(hf_dir / "model_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
        tokenizer.save_pretrained(hf_dir)
        checkpoint_history.append(optimizer_step)
        if save_total_limit is not None and save_total_limit > 0:
            checkpoint_history.sort()
            while len(checkpoint_history) > save_total_limit:
                old_step = checkpoint_history.pop(0)
                old_path = out_dir / f"checkpoint-{old_step}-{model_tag}"
                if old_path.is_dir():
                    shutil.rmtree(old_path)
                    logger.info("Removed oldest checkpoint directory: %s", old_path)
    accelerator.wait_for_everyone()


def _load_legacy_revision_checkpoint(
    accelerator: Accelerator,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ckpt_path: Path,
) -> None:
    unwrapped = accelerator.unwrap_model(model)
    map_loc = accelerator.device
    unwrapped.load_state_dict(torch.load(ckpt_path / "pytorch_model.bin", map_location=map_loc))
    optimizer.load_state_dict(torch.load(ckpt_path / "optimizer.pt", map_location=map_loc))
    scheduler.load_state_dict(torch.load(ckpt_path / "scheduler.pt", map_location=map_loc))

def train_one_model(
    args,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    tokenizer,
) -> Dict:
    model_tag = "krause"
    use_cuda = torch.cuda.is_available() and args.device == "cuda"

    if use_cuda and not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            logger.info("TF32: cuda.matmul.allow_tf32=True, cudnn.allow_tf32=True")

    cfg = ModelConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        max_seq_len=args.block_size,
        window_size=args.window_size,
        top_k=args.top_k,
        init_sigma=args.init_sigma,
        init_standard_weight=args.init_standard_weight,
        gradient_checkpointing=(not use_cuda) and (not args.no_gradient_checkpointing),
        attn_dropout=args.attn_dropout,
        resid_dropout=args.resid_dropout,
        emb_dropout=args.emb_dropout,
    )
    model = CausalTransformerLM(cfg)
    param_count = count_parameters(model)

    if args.compile:
        if not use_cuda:
            raise RuntimeError("--compile is supported on CUDA only (same behavior as the FSDP training script).")
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile requires PyTorch 2.0+.")
        logger.info("torch.compile: wrapping model before Accelerate/FSDP prepare().")
        model = torch.compile(model)

    seq = args.block_size
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    tokens_per_step_target = args.tokens_per_batch
    per_device = max(1, tokens_per_step_target // (seq * world_size))
    actual_tokens = per_device * seq * world_size
    grad_accum = max(1, math.ceil(tokens_per_step_target / actual_tokens))
    actual_tokens = per_device * seq * world_size * grad_accum

    max_steps = max(1, int(args.total_train_tokens // actual_tokens))
    if args.max_train_steps > 0:
        max_steps = min(max_steps, args.max_train_steps)

    warmup_steps = int(max_steps * args.warmup_ratio)

    fsdp_plugin = build_fsdp_plugin_for_revision() if use_cuda else None
    mixed_precision = "bf16" if use_cuda else "no"
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=grad_accum,
        fsdp_plugin=fsdp_plugin,
    )

    if accelerator.is_main_process:
        logger.info(
            "Accelerate+FSDP: grad_accum=%s mixed_precision=%s fsdp=%s",
            grad_accum,
            mixed_precision,
            fsdp_plugin is not None,
        )
        logger.info(
            "gradient_checkpointing (model-level)=%s; CUDA uses FSDP plugin activation checkpointing",
            cfg.gradient_checkpointing,
        )
        logger.info(
            "Batch config: per_device=%s seq=%s world=%s grad_accum=%s -> tokens/optimizer_step=%s ",
            per_device,
            seq,
            accelerator.num_processes,
            grad_accum,
            actual_tokens,
        )
        logger.info("max_steps=%s total_train_tokens=%s", max_steps, args.total_train_tokens)
        print(f"\n===== Training [{model_tag}] =====")
        print(f"Parameters: {param_count / 1e6:.2f}M")

    collator = RevisionLMDataCollator()
    dl_kw = data_loader_kwargs(args)
    train_loader = DataLoader(
        train_ds,
        batch_size=per_device,
        shuffle=True,
        collate_fn=collator,
        **dl_kw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=per_device,
        shuffle=False,
        collate_fn=collator,
        **dl_kw,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=per_device,
        shuffle=False,
        collate_fn=collator,
        **dl_kw,
    )

    _adam_kw: Dict[str, Any] = {
        "lr": args.peak_lr,
        "weight_decay": args.weight_decay,
        "betas": (0.9, 0.95),
    }
    if use_cuda and not args.no_fused_adamw:
        try:
            optimizer = torch.optim.AdamW(model.parameters(), **_adam_kw, fused=True)
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                logger.info("AdamW fused=True (CUDA)")
        except (TypeError, RuntimeError) as e:
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                logger.warning("AdamW fused is unavailable, falling back to the default implementation: %s", e)
            optimizer = torch.optim.AdamW(model.parameters(), **_adam_kw)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **_adam_kw)
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
        num_cycles=0.5,
        min_lr_rate=args.min_lr_ratio,
    )

    out_dir = Path(args.output_root) / revision_model_subdir(args)
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )
    accelerator.register_for_checkpointing(scheduler)

    resume_step = 0
    latest_ck = find_latest_revision_checkpoint(out_dir, model_tag)
    if latest_ck is not None:
        resume_step, ckpt_path = latest_ck
        if accelerator.is_main_process:
            logger.info("Resuming from checkpoint %s (optimizer_step=%s)", ckpt_path, resume_step)
        loaded = False
        try:
            accelerator.load_state(str(ckpt_path))
            loaded = True
        except Exception as e:
            if accelerator.is_main_process:
                logger.warning("accelerator.load_state failed (%s); trying legacy pytorch_model.bin checkpoint", e)
        if not loaded and (ckpt_path / "pytorch_model.bin").is_file():
            _load_legacy_revision_checkpoint(accelerator, model, optimizer, scheduler, ckpt_path)
            loaded = True
        if not loaded:
            resume_step = 0
            if accelerator.is_main_process:
                logger.warning("Checkpoint loading failed; training will restart from step 0.")

    eval_log_path = out_dir / "eval_losses.jsonl"
    metrics_path = out_dir / "final_metrics.json"
    checkpoint_history: List[int] = list_revision_checkpoint_steps(out_dir, model_tag)
    last_val_nll: Optional[float] = None

    train_len = max(1, len(train_loader))
    batches_to_skip = (resume_step * grad_accum) % train_len

    def get_infinite_loader(dl: DataLoader, skip: int):
        while True:
            iter_dl = iter(dl)
            if skip > 0:
                if accelerator.is_main_process:
                    logger.info(
                        "Skipping first %s batches in DataLoader to align with checkpoint...",
                        skip,
                    )
                for _ in range(skip):
                    next(iter_dl, None)
                skip = 0
            for b in iter_dl:
                yield b

    train_iter = get_infinite_loader(train_loader, batches_to_skip)
    optimizer_step = resume_step
    tokens_seen = resume_step * actual_tokens
    pbar = tqdm(total=max_steps, initial=resume_step, disable=not accelerator.is_main_process)
    model.train()

    while optimizer_step < max_steps:
        batch = {k: v.to(accelerator.device) for k, v in next(train_iter).items()}
        with accelerator.accumulate(model):
            loss = compute_revision_lm_loss(model, batch)
            accelerator.backward(loss)
            accum_loss = loss.detach().float()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        tokens_seen += batch["input_ids"].numel() * accelerator.num_processes

        if accelerator.sync_gradients:
            scheduler.step()
            optimizer_step += 1
            pbar.update(1)

            if accelerator.is_main_process and optimizer_step % args.log_every_steps == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    "step=%s loss=%.4f lr=%.2e tokens_seen=%s",
                    optimizer_step,
                    float(accum_loss),
                    lr,
                    tokens_seen,
                )

            if optimizer_step % args.eval_every_steps == 0 or optimizer_step == max_steps:
                val_loss = eval_revision_loss_accelerate(
                    model, val_loader, accelerator, max_eval_steps=args.max_eval_steps
                )
                last_val_nll = val_loss
                if accelerator.is_main_process:
                    record = {
                        "step": optimizer_step,
                        "val_loss": val_loss,
                        "val_ppl": math.exp(min(20.0, val_loss)),
                    }
                    with open(eval_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                    pbar.set_postfix({"vloss": f"{val_loss:.4f}"})
                model.train()

            if (
                args.save_every_steps > 0
                and optimizer_step > 0
                and optimizer_step % args.save_every_steps == 0
            ):
                save_revision_rotating_checkpoint(
                    accelerator,
                    model,
                    tokenizer,
                    cfg,
                    out_dir,
                    optimizer_step,
                    checkpoint_history,
                    args.save_total_limit,
                    model_tag,
                )

    pbar.close()

    test_nll = eval_revision_loss_accelerate(
        model, test_loader, accelerator, max_eval_steps=args.max_eval_steps
    )
    test_ppl = float(math.exp(min(20.0, test_nll)))
    val_nll = last_val_nll if last_val_nll is not None else test_nll
    val_ppl = math.exp(min(20.0, val_nll))

    if accelerator.is_main_process:
        final_metrics = {
            "test_loss": test_nll,
            "test_ppl": test_ppl,
            "max_steps": max_steps,
            "tokens_per_batch": actual_tokens,
            "model_type": model_tag,
            "val_loss": val_nll,
            "val_ppl": val_ppl,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2)
        with open(eval_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": "test", "test_loss": test_nll, "test_ppl": test_ppl}) + "\n")
        print(f"[{model_tag}] TEST loss={test_nll:.4f} ppl={test_ppl:.2f}")

    save_dir = str(out_dir)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(out_dir / f"checkpoint-accelerate-{model_tag}"))
    hf_final = out_dir / f"hf_model-{model_tag}"
    accelerator.save_model(model, hf_final)
    if accelerator.is_main_process:
        with open(hf_final / "model_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
        tokenizer.save_pretrained(hf_final)
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_type": model_tag,
                    "num_parameters": param_count,
                    "last_eval_loss": last_val_nll,
                    "val_nll": val_nll,
                    "val_ppl": val_ppl,
                    "test_nll": test_nll,
                    "test_ppl": test_ppl,
                    "max_steps": max_steps,
                    "tokens_per_optimizer_step": actual_tokens,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[{model_tag}] training state: {out_dir / f'checkpoint-accelerate-{model_tag}'}")
        print(f"[{model_tag}] HF bundle: {hf_final}")

    accelerator.wait_for_everyone()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_type": model_tag,
        "num_parameters": param_count,
        "val_nll": val_nll,
        "val_ppl": val_ppl,
        "test_nll": test_nll,
        "test_ppl": test_ppl,
        "save_dir": save_dir,
    }


def main():
    if "TORCH_DISTRIBUTED_TIMEOUT" not in os.environ:
        os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "7200"
    parser = argparse.ArgumentParser(
        description="Train from scratch ~100M Krause attention language model on ./datasets/fwe10bt."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--train_glob", type=str, default="train.parquet")
    parser.add_argument("--val_glob", type=str, default="val.parquet")
    parser.add_argument("--test_glob", type=str, default="test.parquet")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument(
        "--packed_cache_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--force_repack",
        action="store_true",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./llm/gpt2",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./models_100m",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
    )

    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--ffn_hidden_size", type=int, default=2048)

    parser.add_argument(
        "--block_size",
        type=int,
        default=1024,
        help="Sequence length.",
    )
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument(
        "--top_k",
        type=int,
        default=16,
    )
    parser.add_argument("--init_sigma", type=float, default=2.5)
    parser.add_argument(
        "--init_standard_weight",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--total_train_tokens",
        type=int,
        default=int(10e9),
    )
    parser.add_argument(
        "--tokens_per_batch",
        type=int,
        default=500_000,
    )
    parser.add_argument("--peak_lr", type=float, default=6e-4)
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument("--max_train_steps", type=int, default=-1, help="When >0, override max_steps computed from total_train_tokens.")
    parser.add_argument("--max_eval_steps", type=int, default=-1)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--no_tf32",
        action="store_true",
    )
    parser.add_argument(
        "--no_fused_adamw",
        action="store_true",
    )
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--log_every_steps", type=int, default=10)
    parser.add_argument("--eval_every_steps", type=int, default=500)
    parser.add_argument("--save_every_steps", type=int, default=1000)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--resid_dropout", type=float, default=0.1)
    parser.add_argument("--emb_dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--compile",
        action="store_true",
    )
    args = parser.parse_args()

    use_ddp, rank, world_size = init_distributed()
    is_main = rank == 0
    accelerate_set_seed(args.seed + int(os.environ.get("RANK", "0")))
    if is_main:
        os.makedirs(args.output_root, exist_ok=True)
    if use_ddp:
        dist.barrier()

    if is_main:
        print(
            f"Distributed init (barrier/data sync): use_ddp={use_ddp}, rank/world_size={rank}/{world_size}. "
        )
        print(
            f"Training schedule: total_train_tokens={args.total_train_tokens}, "
            f"tokens_per_batch={args.tokens_per_batch}, peak_lr={args.peak_lr}, warmup_ratio={args.warmup_ratio}"
        )
        print(
            f"log_every/eval_every/save_every: {args.log_every_steps}/{args.eval_every_steps}/{args.save_every_steps} "
            f"(save_total_limit={args.save_total_limit})"
        )
        print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.model_max_length = max(args.block_size, 1_000_000)

    data_dir = Path(args.data_dir).expanduser().resolve()
    packed_cache_root = (
        Path(args.packed_cache_dir).expanduser().resolve()
        if args.packed_cache_dir
        else data_dir / "packed_tokenized"
    )

    if is_main:
        print(f"Loading text splits from data_dir={data_dir} ...")
    train_raw, val_raw, test_raw = load_text_splits(
        data_dir,
        args.train_glob,
        args.val_glob,
        args.test_glob,
        args.text_column,
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
    )

    train_ds = load_or_tokenize_packed(
        train_raw,
        tokenizer,
        args.block_size,
        "train",
        packed_cache_root,
        args.force_repack,
        num_proc=args.num_proc,
    )
    val_ds = load_or_tokenize_packed(
        val_raw,
        tokenizer,
        args.block_size,
        "val",
        packed_cache_root,
        args.force_repack,
        num_proc=args.num_proc,
    )
    test_ds = load_or_tokenize_packed(
        test_raw,
        tokenizer,
        args.block_size,
        "test",
        packed_cache_root,
        args.force_repack,
        num_proc=args.num_proc,
    )

    if is_main:
        print(f"Packed cache root: {packed_cache_root}")
        print(f"Train chunks={len(train_ds)}, Val chunks={len(val_ds)}, Test chunks={len(test_ds)}")
    if use_ddp:
        dist.barrier()

    all_metrics = []
    metrics = train_one_model(
        args,
        train_ds,
        val_ds,
        test_ds,
        tokenizer,
    )
    if is_main:
        all_metrics.append(metrics)

    summary_name = "summary_metrics_100m_krause.json"
    summary_path = os.path.join(args.output_root, summary_name)
    if is_main:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"Done. Summary saved to: {summary_path}")

    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
