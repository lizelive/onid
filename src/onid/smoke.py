from __future__ import annotations

import argparse
import json
from pathlib import Path

from onid.pairs import precompute_pairs
from onid.train import train_experiment


def run_smoke(
    output_root: str | Path,
    train_samples: int,
    val_samples: int,
    epochs: int,
    batch_size: int,
    embedding_kind: str,
    resolution: int,
    shard_size: int,
    streaming: bool,
) -> Path:
    output_root = Path(output_root)
    train_pairs = output_root / "pairs-train"
    val_pairs = output_root / "pairs-val"
    run_dir = output_root / "run"
    output_root.mkdir(parents=True, exist_ok=True)

    precompute_pairs(
        output_dir=train_pairs,
        split="train",
        embedding_kind=embedding_kind,
        resolution=resolution,
        max_samples=train_samples,
        shard_size=shard_size,
        streaming=streaming,
    )
    precompute_pairs(
        output_dir=val_pairs,
        split="validation",
        embedding_kind=embedding_kind,
        resolution=resolution,
        max_samples=val_samples,
        shard_size=shard_size,
        streaming=streaming,
    )
    metrics_path = train_experiment(
        train_dir=train_pairs,
        val_dir=val_pairs,
        output_dir=run_dir,
        embedding_kind=embedding_kind,
        epochs=epochs,
        batch_size=batch_size,
    )

    summary = {
        "train_manifest": str(train_pairs / "manifest.json"),
        "val_manifest": str(val_pairs / "manifest.json"),
        "metrics": json.loads(metrics_path.read_text()),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the smoke-scale ONID experiment end to end")
    parser.add_argument("--output-root", default="outputs/smoke_e2e")
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--val-samples", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embedding-kind", default="pooled", choices=["pooled", "dense"])
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--streaming", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run_smoke(
        output_root=args.output_root,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_kind=args.embedding_kind,
        resolution=args.resolution,
        shard_size=args.shard_size,
        streaming=args.streaming,
    )


if __name__ == "__main__":
    main()
