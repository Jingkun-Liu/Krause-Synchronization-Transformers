from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterator, List, Tuple
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

def _iter_text_rows_shuffled(files: List[Path], rng: random.Random) -> Iterator[Tuple[str, int]]:
    files = list(files)
    rng.shuffle(files)
    
    error_log_path = Path("corrupt_files.log")
    
    for fp in files:
        try:
            pf = pq.ParquetFile(fp)
            for batch in pf.iter_batches(batch_size=65536, columns=["text", "token_count"]):
                tbl = pa.Table.from_batches([batch])
                n = tbl.num_rows
                if n == 0: continue
                
                idx = list(range(n))
                rng.shuffle(idx)
                text_col = tbl.column("text")
                tc_col = tbl.column("token_count")
                
                for i in idx:
                    text = text_col[i].as_py()
                    tc = int(tc_col[i].as_py())
                    if text and tc > 0:
                        yield str(text), tc
                        
        except Exception as e:
            error_msg = f"\n[ERROR] File corrupt: {fp}\nReason: {e}\n"
            print(error_msg)
            
            with open(error_log_path, "a") as f:
                f.write(f"{fp}\n")
            
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fwe10bt train/val/test parquet from FineWeb sample.")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--total_tokens", type=int, default=10_000_000_000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--buffer_rows", type=int, default=50_000)
    args = parser.parse_args()

    files = sorted(args.source.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No *.parquet under {args.source}")

    args.out.mkdir(parents=True, exist_ok=True)
    out_paths = {
        "train": args.out / "train.parquet",
        "val":   args.out / "val.parquet",
        "test":  args.out / "test.parquet",
    }

    schema = pa.schema([("text", pa.string())])
    rng = random.Random(args.seed)
    row_iter = _iter_text_rows_shuffled(files, rng)

    writers: Dict[str, pq.ParquetWriter | None] = {"train": None, "val": None, "test": None}
    buffers: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    used = {"train": 0, "val": 0, "test": 0}
    total_used = 0

    def flush(name: str) -> None:
        if not buffers[name]: return
        t = pa.Table.from_pylist(buffers[name], schema=schema)
        if writers[name] is None:
            writers[name] = pq.ParquetWriter(str(out_paths[name]), schema)
        writers[name].write_table(t)
        buffers[name].clear()

    pbar = tqdm(total=args.total_tokens, unit="tok", desc="Sampling") if tqdm else None

    splits = ["train", "val", "test"]
    weights = [500, 1, 1]

    for text, tc in row_iter:
        if total_used >= args.total_tokens:
            break

        target_split = rng.choices(splits, weights=weights, k=1)[0]
        
        buffers[target_split].append({"text": text})
        used[target_split] += tc
        total_used += tc
        
        if pbar: pbar.update(tc)

        if len(buffers[target_split]) >= args.buffer_rows:
            flush(target_split)

    for name in splits:
        flush(name)
        if writers[name]:
            writers[name].close()
    if pbar: pbar.close()

    print("\nExtraction Complete!")
    print(f"Target Total Tokens: {args.total_tokens:,}")
    print(f"Actual Tokens Used:  train={used['train']:,}, val={used['val']:,}, test={used['test']:,}")
    print(f"Total Combined:      {total_used:,}")

if __name__ == "__main__":
    main()
