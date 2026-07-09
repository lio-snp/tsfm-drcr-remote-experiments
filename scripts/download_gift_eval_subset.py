#!/usr/bin/env python
"""Download a narrow GIFT-Eval dataset subset from Hugging Face."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def print_status(status: dict[str, object]) -> None:
    print(json.dumps(status, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Salesforce/GiftEval")
    parser.add_argument("--dataset-name", default="bizitobs_application")
    parser.add_argument("--local-dir", default="data/gift-eval")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = local_dir / args.dataset_name
    try:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=[f"{args.dataset_name}/*"],
            local_dir=str(local_dir),
        )
    except Exception as exc:  # noqa: BLE001 - downloader should emit a reusable status
        print_status(
            {
                "status": "blocked_download_failed",
                "timestamp": int(time.time()),
                "repo_id": args.repo_id,
                "dataset_name": args.dataset_name,
                "local_dir": str(local_dir.resolve()),
                "dataset_dir": str(dataset_dir.resolve()),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise SystemExit(1) from exc

    if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
        print_status(
            {
                "status": "blocked_download_failed",
                "timestamp": int(time.time()),
                "repo_id": args.repo_id,
                "dataset_name": args.dataset_name,
                "local_dir": str(local_dir.resolve()),
                "dataset_dir": str(dataset_dir.resolve()),
                "error": "snapshot_download returned without creating the requested dataset directory",
            }
        )
        raise SystemExit(1)

    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "repo_id": args.repo_id,
        "dataset_name": args.dataset_name,
        "local_dir": str(local_dir.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
    }
    print_status(status)


if __name__ == "__main__":
    main()
