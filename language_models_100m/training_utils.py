import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def init_distributed() -> Tuple[bool, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        return True, rank, world_size
    return False, 0, 1

def load_text_splits(
    data_dir: Path,
    train_glob: str,
    val_glob: str,
    test_glob: str,
    text_column: str,
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    def _load_split(pattern: str, split_name: str) -> Dataset:
        paths = sorted(data_dir.glob(pattern))
        if not paths:
            fixed = data_dir / f"{split_name}.parquet"
            if fixed.is_file():
                paths = [fixed]
        if not paths:
            raise FileNotFoundError(
                f"No files for {split_name} under {data_dir} (pattern {pattern!r}; "
                f"also tried {split_name}.parquet). Set --data_dir to the folder containing these parquet files."
            )
        ext = paths[0].suffix.lower()
        if ext in (".arrow",):
            if len(paths) != 1:
                raise ValueError(f"Expected single Arrow shard for {split_name}, got {paths}")
            return load_from_disk(str(paths[0]))
        if ext in (".parquet",):
            ds = load_dataset("parquet", data_files=[str(p) for p in paths], split="train")
            return ds
        if ext in (".json", ".jsonl"):
            ds = load_dataset("json", data_files=[str(p) for p in paths], split="train")
            return ds
        raise ValueError(f"Unsupported file type: {paths[0]}")

    def _load_explicit(path: Path, split_name: str) -> Dataset:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            return load_from_disk(str(path))
        ext = path.suffix.lower()
        if ext == ".parquet":
            return load_dataset("parquet", data_files=str(path), split="train")
        if ext in (".json", ".jsonl"):
            return load_dataset("json", data_files=str(path), split="train")
        raise ValueError(f"Unsupported file for {split_name}: {path}")

    if train_path and val_path and test_path:
        train_ds = _load_explicit(Path(train_path), "train")
        val_ds = _load_explicit(Path(val_path), "val")
        test_ds = _load_explicit(Path(test_path), "test")
    else:
        train_ds = _load_split(train_glob, "train")
        val_ds = _load_split(val_glob, "val")
        test_ds = _load_split(test_glob, "test")

    def _ensure_text(ds: Dataset, name: str) -> Dataset:
        if text_column not in ds.column_names:
            if "text" in ds.column_names:
                return ds
            raise KeyError(f"Column {text_column!r} not in {name}: {ds.column_names}")
        if text_column != "text":
            ds = ds.rename_column(text_column, "text")
        return ds

    train_ds = _ensure_text(train_ds, "train")
    val_ds = _ensure_text(val_ds, "val")
    test_ds = _ensure_text(test_ds, "test")
    return train_ds, val_ds, test_ds


def load_or_tokenize_packed(
    ds: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    split_name: str,
    packed_cache_root: Path,
    force_repack: bool,
    num_proc: int = 4,
) -> Dataset:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer.eos_token_id is required for EOS-separated packing.")

    def _pack_batch(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        merged: List[int] = []
        for text in batch["text"]:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            merged.extend(ids)
            merged.append(eos_id)
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        for start in range(0, len(merged), max_length):
            chunk = merged[start : start + max_length]
            if len(chunk) < max_length:
                break
            input_ids.append(chunk)
            attention_mask.append([1] * max_length)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    seq_dir = packed_cache_root / f"seq{max_length}"
    out_dir = seq_dir / split_name
    ready_flag = out_dir / ".ready"
    map_arrow = seq_dir / f".hf_map_{split_name}_seq{max_length}.arrow"

    cache_ok = out_dir.is_dir() and ready_flag.is_file() and not force_repack
    if cache_ok:
        if local_rank == 0:
            logger.info("Loading packed %s from %s", split_name, out_dir)
        return load_from_disk(str(out_dir))

    if local_rank == 0:
        logger.info(
            "Building packed %s (seq=%s) -> %s (single rank; other ranks wait on .ready)",
            split_name,
            max_length,
            out_dir,
        )
        seq_dir.mkdir(parents=True, exist_ok=True)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        if force_repack and map_arrow.exists():
            map_arrow.unlink()

        cols = list(ds.column_names)
        packed = ds.map(
            _pack_batch,
            batched=True,
            batch_size=10_000,
            remove_columns=cols,
            num_proc=num_proc,
            load_from_cache_file=True,
            cache_file_name=str(map_arrow),
            desc="Pack tokenize (concat + chunk)",
        )
        packed.save_to_disk(str(out_dir))
        ready_flag.touch()
    else:
        n = 0
        while not ready_flag.is_file():
            time.sleep(2.0)
            n += 1
            if n % 15 == 0:
                logger.info(
                    "Waiting for packed %s at %s (rank %s, waited ~%ds)",
                    split_name,
                    out_dir,
                    local_rank,
                    n * 2,
                )
    return load_from_disk(str(out_dir))

class RevisionLMDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([x["input_ids"] for x in features], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def compute_revision_lm_loss(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    input_ids = batch["input_ids"]
    logits = model(input_ids[:, :-1])
    labels = input_ids[:, 1:]
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="mean",
    )


def eval_revision_loss_accelerate(
    model: nn.Module,
    dataloader: DataLoader,
    accelerator: Accelerator,
    max_eval_steps: int = -1,
) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if max_eval_steps > 0 and step >= max_eval_steps:
                break
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            with torch.autocast(device_type=accelerator.device.type, dtype=torch.bfloat16, enabled=True):
                loss = compute_revision_lm_loss(model, batch)
            losses.append(accelerator.gather(loss.detach().unsqueeze(0)).mean().item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def revision_model_subdir(args) -> str:
    return f"krause_sigma_{args.init_sigma:g}"


def find_latest_revision_checkpoint(out_dir: Path, model_tag: str) -> Optional[Tuple[int, Path]]:
    if not out_dir.is_dir():
        return None
    prefix = "checkpoint-"
    suffix = f"-{model_tag}"
    best_step = -1
    best_path: Optional[Path] = None
    for p in out_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        mid = name[len(prefix) : -len(suffix)]
        if not mid.isdigit():
            continue
        step = int(mid)
        if step > best_step:
            best_step = step
            best_path = p
    if best_path is None:
        return None
    return (best_step, best_path)


def list_revision_checkpoint_steps(out_dir: Path, model_tag: str) -> List[int]:
    if not out_dir.is_dir():
        return []
    prefix = "checkpoint-"
    suffix = f"-{model_tag}"
    steps: List[int] = []
    for p in out_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        mid = name[len(prefix) : -len(suffix)]
        if mid.isdigit():
            steps.append(int(mid))
    steps.sort()
    return steps


def data_loader_kwargs(args) -> Dict[str, Any]:
    kw: Dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = args.prefetch_factor
    return kw
