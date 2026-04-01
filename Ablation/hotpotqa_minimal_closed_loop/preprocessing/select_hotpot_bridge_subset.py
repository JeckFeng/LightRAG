from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from Ablation.hotpotqa_minimal_closed_loop.common import (
    DEFAULT_HOTPOT_DATASET_PATH,
    EXPERIMENT_ROOT,
    build_case_manifest,
    filter_bridge_samples,
    load_hotpot_dataset,
    read_json,
    select_bridge_subset,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a deterministic HotpotQA bridge subset and export a JSONL manifest."
    )
    parser.add_argument("--config", type=Path, help="Subset config JSON path.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_HOTPOT_DATASET_PATH,
        help="Path to hotpot_dev_distractor_v1.json.",
    )
    parser.add_argument("--subset-id", type=str, help="Subset identifier such as bridge_100.")
    parser.add_argument("--limit", type=int, help="Maximum number of bridge samples.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle bridge samples deterministically.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic shuffle seed.")
    parser.add_argument("--output", type=Path, help="Output manifest path.")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if args.config:
        config = read_json(args.config)

    dataset_path = Path(config.get("dataset_path", args.dataset))
    subset_id = args.subset_id or config.get("subset_id")
    limit = args.limit if args.limit is not None else config.get("sample_limit")
    shuffle = args.shuffle or bool(config.get("shuffle", False))
    seed = args.seed if args.seed is not None else int(config.get("seed", 0))
    output_path = args.output or config.get("output_manifest")

    if subset_id is None:
        raise ValueError("subset_id is required via --subset-id or config file")

    if output_path is None:
        output_path = (
            EXPERIMENT_ROOT / "manifests" / f"{subset_id}_case_manifest.jsonl"
        )
    else:
        output_path = Path(output_path)

    return {
        "dataset_path": dataset_path,
        "subset_id": subset_id,
        "limit": limit,
        "shuffle": shuffle,
        "seed": seed,
        "output_path": output_path,
    }


def build_manifest_rows(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = load_hotpot_dataset(config["dataset_path"])
    bridge_samples = filter_bridge_samples(samples)
    selected_samples = select_bridge_subset(
        bridge_samples,
        config["limit"],
        shuffle=config["shuffle"],
        seed=config["seed"],
    )

    manifest_rows = [
        build_case_manifest(sample, order_index=index).to_dict()
        for index, sample in enumerate(selected_samples)
    ]
    summary = {
        "subset_id": config["subset_id"],
        "dataset_path": str(config["dataset_path"]),
        "bridge_total": len(bridge_samples),
        "selected_count": len(manifest_rows),
        "shuffle": config["shuffle"],
        "seed": config["seed"],
        "output_manifest": str(config["output_path"]),
    }
    return manifest_rows, summary


def export_bridge_subset(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_rows, summary = build_manifest_rows(config)
    write_jsonl(config["output_path"], manifest_rows)
    summary_path = config["output_path"].with_suffix(".summary.json")
    write_json(summary_path, summary)
    return manifest_rows, summary


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    manifest_rows, _summary = export_bridge_subset(config)

    print(
        f"[select_hotpot_bridge_subset] subset={config['subset_id']} "
        f"selected={len(manifest_rows)} output={config['output_path']}"
    )


if __name__ == "__main__":
    main()
