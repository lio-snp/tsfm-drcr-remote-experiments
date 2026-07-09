#!/usr/bin/env python
"""Convert traffic HDF5/CSV/NPY data into a Chronos-env friendly NPZ matrix."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from low_snr_tsfm.traffic import load_traffic_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", required=True)
    args = parser.parse_args()

    input_path = ROOT / args.input
    output_path = ROOT / args.output
    matrix = load_traffic_matrix(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, data=matrix)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "timestamp": int(time.time()),
                "dataset_name": args.dataset_name,
                "input": str(input_path),
                "output": str(output_path),
                "shape": list(matrix.shape),
                "format": "time x sensor traffic speed matrix",
            },
            indent=2,
        )
    )
    print(json.dumps({"output": str(output_path), "shape": list(matrix.shape)}, indent=2))


if __name__ == "__main__":
    main()
