from __future__ import annotations
import argparse
import glob
import importlib.util
import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_REPO_ROOT = Path(__file__).resolve().parent

def dist_setup() -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size

def dist_sum_two_ints(a: int, b: int, device: torch.device, use_dist: bool) -> Tuple[int, int]:
    if not use_dist:
        return a, b
    t = torch.tensor([a, b], dtype=torch.long, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t[0].item()), int(t[1].item())

def _import_train():
    spec = importlib.util.spec_from_file_location(
        "train_100m",
        str(_REPO_ROOT / "train_100m.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CausalTransformerLM, mod.ModelConfig

CausalTransformerLM, ModelConfig = _import_train()

def model_config_from_json(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    valid = {f.name for f in fields(ModelConfig)}
    kwargs = {k: v for k, v in raw.items() if k in valid}
    return ModelConfig(**kwargs)


def _load_state_dict(model_dir: Path) -> Dict[str, torch.Tensor]:
    model_dir = model_dir.resolve()
    weights = model_dir / "model.safetensors"
    if weights.is_file():
        from safetensors.torch import load_file
        return load_file(str(weights))
    pt = model_dir / "pytorch_model.bin"
    if pt.is_file():
        return torch.load(str(pt), map_location="cpu")
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin under {model_dir}")


def load_revision_checkpoint(
    model_path: str,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[nn.Module, ModelConfig]:
    path = Path(model_path).expanduser().resolve()
    cfg = model_config_from_json(path / "model_config.json")
    model = CausalTransformerLM(cfg)
    state = _load_state_dict(path)
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, cfg


@torch.inference_mode()
def seq_log_prob_sum(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_start: int,
) -> float:
    logits = model(input_ids).float()
    logits = logits[0]
    log_probs = F.log_softmax(logits, dim=-1)
    score = 0.0
    for i in range(target_start, input_ids.shape[1]):
        tid = input_ids[0, i].item()
        score += log_probs[i - 1, tid].item()
    return score


@torch.inference_mode()
def score_continuation_sum(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    context: str,
    continuation: str,
    device: torch.device,
    max_length: int,
) -> float:
    cont_text = continuation if continuation.startswith(" ") else " " + continuation
    full_text = context + cont_text
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    if len(full_ids) >= len(ctx_ids) and full_ids[: len(ctx_ids)] == ctx_ids:
        target_start = len(ctx_ids)
    else:
        cont_ids = tokenizer.encode(cont_text, add_special_tokens=False)
        full_ids = ctx_ids + cont_ids
        target_start = len(ctx_ids)
    if len(full_ids) > max_length:
        overflow = len(full_ids) - max_length
        full_ids = full_ids[overflow:]
        target_start = max(0, target_start - overflow)
    if target_start >= len(full_ids):
        return float("-inf")
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    return seq_log_prob_sum(model, input_ids, target_start)


@torch.inference_mode()
def score_full_sequence_sum_from(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    text: str,
    device: torch.device,
    max_length: int,
    target_start: int,
) -> float:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < 2:
        return float("-inf")
    if len(ids) > max_length:
        ids = ids[-max_length:]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    return seq_log_prob_sum(model, input_ids, target_start)


def eval_hellaswag(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    data_root: Path,
    split: str,
    device: torch.device,
    max_length: int,
    rank: int = 0,
    world_size: int = 1,
    use_dist: bool = False,
    show_pbar: bool = True,
) -> Dict[str, Any]:
    sub = "validation" if split == "validation" else "test"
    files = sorted(glob.glob(str(data_root / "datasets" / "hellaswag" / "data" / f"{sub}-*.parquet")))
    ds = load_dataset("parquet", data_files=files, split="train")
    n_examples = len(ds)
    n_ok = 0
    n_total = 0
    for idx, row in enumerate(tqdm(ds, desc=f"HellaSwag ({split})", disable=not show_pbar)):
        if idx % world_size != rank:
            continue
        ctx = row["ctx"]
        endings = row["endings"]
        lp = [
            score_continuation_sum(model, tokenizer, ctx, str(e), device, max_length) for e in endings
        ]
        pred = int(np.argmax(lp))
        gold = int(str(row["label"]).strip())
        n_total += 1
        n_ok += int(pred == gold)
    n_ok, n_total = dist_sum_two_ints(n_ok, n_total, device, use_dist)
    return {
        "dataset": "hellaswag",
        "split": split,
        "n_examples": n_examples,
        "n": n_total,
        "correct": n_ok,
        "accuracy": n_ok / max(n_total, 1),
    }


def eval_piqa(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    data_root: Path,
    split: str,
    device: torch.device,
    max_length: int,
    rank: int = 0,
    world_size: int = 1,
    use_dist: bool = False,
    show_pbar: bool = True,
) -> Dict[str, Any]:
    name = "piqa_validation.parquet" if split == "validation" else "piqa_test.parquet"
    path = data_root / "datasets" / "piqa" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    ds = load_dataset("parquet", data_files=str(path), split="train")
    n_examples = len(ds)
    n_ok = 0
    n_total = 0
    for idx, row in enumerate(tqdm(ds, desc=f"PIQA ({split})", disable=not show_pbar)):
        if idx % world_size != rank:
            continue
        goal = str(row["goal"]).strip()
        s1, s2 = row["sol1"], row["sol2"]
        lp = [
            score_continuation_sum(model, tokenizer, goal, str(s1), device, max_length),
            score_continuation_sum(model, tokenizer, goal, str(s2), device, max_length),
        ]
        pred = int(np.argmax(lp))
        gold = int(row["label"])
        n_total += 1
        n_ok += int(pred == gold)
    n_ok, n_total = dist_sum_two_ints(n_ok, n_total, device, use_dist)
    return {
        "dataset": "piqa",
        "split": split,
        "n_examples": n_examples,
        "n": n_total,
        "correct": n_ok,
        "accuracy": n_ok / max(n_total, 1),
    }


def eval_blimp(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    data_root: Path,
    device: torch.device,
    max_length: int,
    rank: int = 0,
    world_size: int = 1,
    use_dist: bool = False,
    show_pbar: bool = True,
) -> Dict[str, Any]:
    files = sorted(glob.glob(str(data_root / "datasets" / "blimp" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(data_root / "datasets/blimp/**/*.parquet")
    ds = load_dataset("parquet", data_files=files, split="train")
    n_examples = len(ds)
    n_ok = 0
    n_total = 0
    for idx, row in enumerate(tqdm(ds, desc="BLiMP (all)", disable=not show_pbar)):
        if idx % world_size != rank:
            continue
        good = row["sentence_good"].strip()
        bad = row["sentence_bad"].strip()
        s_good = score_full_sequence_sum_from(model, tokenizer, good, device, max_length, 1)
        s_bad = score_full_sequence_sum_from(model, tokenizer, bad, device, max_length, 1)
        lp = [s_good, s_bad]
        pred = int(np.argmax(lp))
        n_total += 1
        n_ok += int(pred == 0)
    n_ok, n_total = dist_sum_two_ints(n_ok, n_total, device, use_dist)
    return {
        "dataset": "blimp",
        "split": "all_parquet",
        "n": n_total,
        "n_examples": n_examples,
        "correct": n_ok,
        "accuracy": n_ok / max(n_total, 1),
    }


def _parse_arc_choices_and_gold(row: Dict[str, Any]) -> Tuple[List[str], int]:
    texts: List[str]
    ch = row.get("choices")
    if ch is not None:
        if isinstance(ch, dict) and "text" in ch:
            texts = [str(x) for x in ch["text"]]
        elif isinstance(ch, (list, tuple)):
            texts = []
            for c in ch:
                if isinstance(c, str):
                    texts.append(c)
                elif isinstance(c, dict) and "text" in c:
                    texts.append(str(c["text"]))
                else:
                    texts.append(str(c))
        else:
            texts = [str(ch)]
    elif all(k in row for k in ("A", "B", "C", "D")):
        texts = [str(row["A"]), str(row["B"]), str(row["C"]), str(row["D"])]
    else:
        raise KeyError("no choices")

    while len(texts) < 4:
        texts.append("")
    texts = texts[:4]
    n = len(texts)

    g = row.get("answer", row.get("label", row.get("answerKey")))
    if isinstance(g, (list, tuple)) and len(g) == 1:
        g = g[0]
    if isinstance(g, str):
        gs = g.strip().upper()
        if len(gs) == 1 and "A" <= gs <= "D":
            gold = ord(gs) - ord("A")
        else:
            gold = int(gs)
    else:
        gold = int(g)
    gold %= n
    return texts, gold


def _glob_arc_e_parquet(data_root: Path, split: str) -> List[str]:
    base = data_root / "datasets" / "arc_e"
    if split == "all":
        files = sorted(glob.glob(str(base / "**" / "*.parquet"), recursive=True))
    else:
        files = sorted(glob.glob(str(base / f"{split}-*.parquet")))
    return files


def eval_arc_e(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    data_root: Path,
    device: torch.device,
    max_length: int,
    split: str = "test",
    rank: int = 0,
    world_size: int = 1,
    use_dist: bool = False,
    show_pbar: bool = True,
) -> Dict[str, Any]:
    files = _glob_arc_e_parquet(data_root, split)
    ds = load_dataset("parquet", data_files=files, split="train")
    n_examples = len(ds)
    n_ok = 0
    n_total = 0
    for idx, row in enumerate(tqdm(ds, desc="ARC-E", disable=not show_pbar)):
        if idx % world_size != rank:
            continue
        q = str(row.get("question", row.get("query", ""))).strip()
        texts, gold = _parse_arc_choices_and_gold(dict(row))
        lp = [score_continuation_sum(model, tokenizer, q, str(t), device, max_length) for t in texts]
        pred = int(np.argmax(lp))
        n_total += 1
        n_ok += int(pred == gold)
    n_ok, n_total = dist_sum_two_ints(n_ok, n_total, device, use_dist)
    return {
        "dataset": "arc_e",
        "split": split,
        "n": n_total,
        "n_examples": n_examples,
        "correct": n_ok,
        "accuracy": n_ok / max(n_total, 1),
    }


def _cbt_answer_to_gold_index(ans: Any, candidates: List[str]) -> int:
    if isinstance(ans, (list, tuple)) and len(ans) == 1:
        ans = ans[0]
    if isinstance(ans, (int, np.integer)):
        i = int(ans)
        if 0 <= i < len(candidates):
            return i
    if isinstance(ans, str) and ans.strip().isdigit():
        i = int(ans.strip())
        if 0 <= i < len(candidates):
            return i
    s = str(ans).strip()
    s_low = s.lower()
    for i, c in enumerate(candidates):
        if c.lower() == s_low:
            return i
    raise ValueError(f"answer {ans!r} not found in candidates")


def _filter_cbt_task_parquets(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        norm = p.replace("\\", "/")
        if "/cbt/raw/" in norm or norm.rstrip("/").endswith("/cbt/raw"):
            continue
        out.append(p)
    return sorted(out)


def _glob_cbt_parquet(data_root: Path, split: str) -> List[str]:
    base = data_root / "datasets" / "cbt"
    if split == "all":
        files = sorted(glob.glob(str(base / "**" / "*.parquet"), recursive=True))
    else:
        files = sorted(glob.glob(str(base / "**" / f"{split}-*.parquet"), recursive=True))
    return _filter_cbt_task_parquets(files)


def eval_cbt(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    data_root: Path,
    device: torch.device,
    max_length: int,
    split: str = "test",
    rank: int = 0,
    world_size: int = 1,
    use_dist: bool = False,
    show_pbar: bool = True,
) -> Dict[str, Any]:
    files = _glob_cbt_parquet(data_root, split)
    if not files:
        raise FileNotFoundError(
            f"{data_root}/datasets/cbt/**/{split}-*.parquet after excluding cbt/raw (or --cbt_split all)"
        )
    ds = load_dataset("parquet", data_files=files, split="train")
    n_examples = len(ds)
    n_ok = 0
    n_total = 0
    for idx, row in enumerate(tqdm(ds, desc="CBT", disable=not show_pbar)):
        if idx % world_size != rank:
            continue
        question = str(row["question"]).strip()
        left, right = question.split("XXXXX", 1)
        sents = row.get("sentences", [])
        if isinstance(sents, str):
            sents = [sents]
        sentences = [str(s).strip() for s in (sents or []) if str(s).strip()]
        options = row.get("options", row.get("candidates", []))
        if isinstance(options, str):
            options = json.loads(options)
        candidates = [str(o).strip() for o in (options or [])]
        gold = _cbt_answer_to_gold_index(row.get("answer"), candidates)
        prefix = "\n".join(sentences) + "\n" + left
        lp = []
        for opt in candidates:
            continuation = opt.strip() + right
            lp.append(
                score_continuation_sum(model, tokenizer, prefix, continuation, device, max_length)
            )
        pred = int(np.argmax(lp))
        n_total += 1
        n_ok += int(pred == gold)
    n_ok, n_total = dist_sum_two_ints(n_ok, n_total, device, use_dist)
    return {
        "dataset": "cbt",
        "split": split,
        "n": n_total,
        "n_examples": n_examples,
        "correct": n_ok,
        "accuracy": n_ok / max(n_total, 1),
    }


def eval_lambada(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    data_root: Path,
    jsonl_relative: str,
    device: torch.device,
    max_length: int,
    rank: int = 0,
    world_size: int = 1,
    use_dist: bool = False,
    show_pbar: bool = True,
) -> Dict[str, Any]:
    path = (data_root / jsonl_relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    ds = load_dataset("json", data_files=str(path), split="train")
    n_examples = len(ds)
    n_ok = 0
    n_total = 0
    for idx, row in enumerate(tqdm(ds, desc="LAMBADA", disable=not show_pbar)):
        if idx % world_size != rank:
            continue
        text = str(row["text"]).strip()
        context, target = text.rsplit(" ", 1)
        target_ids_sp = tokenizer.encode(" " + target, add_special_tokens=False)
        tgt = target_ids_sp if target_ids_sp else tokenizer.encode(target, add_special_tokens=False)
        ctx_ids = tokenizer.encode(context, add_special_tokens=False)
        all_ids = ctx_ids + tgt
        if len(all_ids) > max_length:
            ctx_ids = ctx_ids[-(max_length - len(tgt)) :]
            all_ids = ctx_ids + tgt
        input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(input_ids)
        ctx_len = len(ctx_ids)
        predicted_correctly = True
        for j, tok_id in enumerate(tgt):
            pred = int(torch.argmax(logits[0, ctx_len - 1 + j]).item())
            if pred != tok_id:
                predicted_correctly = False
                break
        n_total += 1
        n_ok += int(predicted_correctly)
    n_ok, n_total = dist_sum_two_ints(n_ok, n_total, device, use_dist)
    return {
        "dataset": "lambada",
        "split": Path(jsonl_relative).stem,
        "file": str(path),
        "n": n_total,
        "n_examples": n_examples,
        "correct": n_ok,
        "accuracy": n_ok / max(n_total, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmarks for train_scratch_revision CausalTransformerLM (hf_model-*/ with model_config.json).",
    )
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Training export directory (hf_model-*/ with model_config.json and weights).",
    )
    p.add_argument(
        "--tokenizer_name",
        type=str,
        default=str(_REPO_ROOT / "llm" / "gpt2"),
        help="Local tokenizer directory (default: repo llm/gpt2).",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default=str(_REPO_ROOT),
        help="Project root containing ./datasets.",
    )
    p.add_argument(
        "--tasks",
        type=str,
        default="hellaswag,piqa,blimp,arc_e,cbt,lambada",
        help="Comma-separated task list.",
    )
    p.add_argument(
        "--arc_e_split",
        type=str,
        default="test",
        choices=["test", "validation", "train", "all"],
        help="ARC-E parquet split (test-*.parquet, etc.); all loads every parquet under arc_e.",
    )
    p.add_argument(
        "--cbt_split",
        type=str,
        default="test",
        choices=["test", "validation", "train", "all"],
        help="CBT split pattern **/test-*.parquet (CN/V/P/NE, etc.); excludes cbt/raw.",
    )
    p.add_argument(
        "--lambada_jsonl",
        type=str,
        default="datasets/lambada/data/lambada_test_en.jsonl",
        help="Path under --data_root to LAMBADA jsonl (field ``text`` per line).",
    )
    p.add_argument(
        "--hellaswag_split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Use validation for labeled accuracy (test has no public labels).",
    )
    p.add_argument(
        "--piqa_split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Use validation for accuracy (test labels are -1).",
    )
    p.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Max sequence length for scoring; capped by model_config.json max_seq_len.",
    )
    p.add_argument(
        "--output_json",
        type=str,
        default="eval_results.json",
        help="Output JSON path.",
    )
    args = p.parse_args()

    use_dist, rank, local_rank, world_size = dist_setup()
    if use_dist:
        device = torch.device(f"cuda:{local_rank}")
        dtype = torch.bfloat16
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    show_pbar = rank == 0

    tok_path = str(Path(args.tokenizer_name).expanduser().resolve())
    tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max(args.max_seq_length, 1_000_000)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    model, rev_cfg = load_revision_checkpoint(args.model_path, dtype, device)
    max_len = min(int(args.max_seq_length), int(rev_cfg.max_seq_len))
    tokenizer.model_max_length = max(max_len, 1_000_000)

    data_root = Path(args.data_root).resolve()
    task_set = {x.strip().lower() for x in args.tasks.split(",") if x.strip()}
    results: Dict[str, Any] = {
        "model_path": str(Path(args.model_path).resolve()),
        "model_type": rev_cfg.model_type,
        "max_seq_len_used": max_len,
    }

    if "hellaswag" in task_set:
        results["hellaswag"] = eval_hellaswag(
            model,
            tokenizer,
            data_root,
            args.hellaswag_split,
            device,
            max_len,
            rank=rank,
            world_size=world_size,
            use_dist=use_dist,
            show_pbar=show_pbar,
        )
        if show_pbar:
            print(json.dumps(results["hellaswag"], indent=2))

    if "piqa" in task_set:
        results["piqa"] = eval_piqa(
            model,
            tokenizer,
            data_root,
            args.piqa_split,
            device,
            max_len,
            rank=rank,
            world_size=world_size,
            use_dist=use_dist,
            show_pbar=show_pbar,
        )
        if show_pbar:
            print(json.dumps(results["piqa"], indent=2))

    if "blimp" in task_set:
        results["blimp"] = eval_blimp(
            model,
            tokenizer,
            data_root,
            device,
            max_len,
            rank=rank,
            world_size=world_size,
            use_dist=use_dist,
            show_pbar=show_pbar,
        )
        if show_pbar:
            print(json.dumps(results["blimp"], indent=2))

    if "arc_e" in task_set or "arc-e" in task_set or "arc_easy" in task_set:
        results["arc_e"] = eval_arc_e(
            model,
            tokenizer,
            data_root,
            device,
            max_len,
            split=args.arc_e_split,
            rank=rank,
            world_size=world_size,
            use_dist=use_dist,
            show_pbar=show_pbar,
        )
        if show_pbar:
            print(json.dumps(results["arc_e"], indent=2))

    if "cbt" in task_set:
        results["cbt"] = eval_cbt(
            model,
            tokenizer,
            data_root,
            device,
            max_len,
            split=args.cbt_split,
            rank=rank,
            world_size=world_size,
            use_dist=use_dist,
            show_pbar=show_pbar,
        )
        if show_pbar:
            print(json.dumps(results["cbt"], indent=2))

    if "lambada" in task_set:
        results["lambada"] = eval_lambada(
            model,
            tokenizer,
            data_root,
            args.lambada_jsonl,
            device,
            max_len,
            rank=rank,
            world_size=world_size,
            use_dist=use_dist,
            show_pbar=show_pbar,
        )
        if show_pbar:
            print(json.dumps(results["lambada"], indent=2))
    if use_dist:
        dist.barrier()

    out_path = Path(args.output_json)
    if show_pbar:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Wrote {out_path.resolve()}")

    if use_dist:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
