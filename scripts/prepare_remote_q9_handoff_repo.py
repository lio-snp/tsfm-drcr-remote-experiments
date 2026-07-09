#!/usr/bin/env python3
"""Prepare a small public-repo handoff for remote q9/full-grid reruns.

The main working tree contains paper drafts, local caches, and large result
artifacts. This script stages only the code, manifests, docs, and contracts
needed by a larger-memory machine to execute the remaining P0 reruns.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "build" / "tsfm-drcr-remote-q9"

SCRIPT_PATTERNS = [
    "*.py",
    "critic_remote_q9_*.sh",
    "critic_oral_*.sh",
    "critic_drcr_paper_readiness_reports.sh",
]

DOC_FILES = [
    "docs/remote_q9_rerun_execution_pack.md",
    "docs/remote_q9_rerun_completion_audit.md",
    "docs/remote_q9_ingestion_manifest.md",
    "docs/aaai_oral_goal_status.md",
    "docs/forecast_artifact_contract.md",
    "docs/metrics.md",
    "docs/paper_main_experiment_assets.md",
]

RESULT_FILES = [
    "results/aaai_stress/remote_q9_rerun_plan.csv",
    "results/aaai_stress/oral_rerun_command_manifest.csv",
    "results/aaai_stress/remote_q9_rerun_completion_audit.csv",
    "results/aaai_stress/remote_q9_ingestion_manifest.csv",
    "results/aaai_stress/oral_evidence_source_gap_matrix.csv",
    "results/aaai_stress/oral_evidence_family_gap_matrix.csv",
]

ROOT_FILES = [
    "pyproject.toml",
    ".gitignore",
]


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, ignore: shutil.IgnorePattern | None = None) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def prepare(out_dir: Path, clean: bool) -> None:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rel in ROOT_FILES:
        copy_file(ROOT / rel, out_dir / rel)

    cache_ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    copy_tree(ROOT / "src", out_dir / "src", ignore=cache_ignore)
    copy_tree(ROOT / "configs", out_dir / "configs")
    copy_tree(ROOT / "tests", out_dir / "tests", ignore=cache_ignore)

    scripts_out = out_dir / "scripts"
    scripts_out.mkdir(parents=True, exist_ok=True)
    for pattern in SCRIPT_PATTERNS:
        for src in (ROOT / "scripts").glob(pattern):
            copy_file(src, scripts_out / src.name)

    for rel in DOC_FILES:
        copy_file(ROOT / rel, out_dir / rel)
    for rel in RESULT_FILES:
        copy_file(ROOT / rel, out_dir / rel)

    copy_tree(ROOT / "results" / "aaai_stress" / "rerun_manifests", out_dir / "results" / "aaai_stress" / "rerun_manifests")
    copy_tree(ROOT / "remote_q9_handoff", out_dir / "remote_q9_handoff")
    copy_file(ROOT / "remote_q9_handoff" / "README.md", out_dir / "README.md")

    (out_dir / "results" / "raw_forecasts").mkdir(parents=True, exist_ok=True)
    (out_dir / "results" / "remote_run_logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "data" / "README.md").write_text(
        "\n".join(
            [
                "# Data Directory",
                "",
                "Place or symlink GIFT-Eval data here using the paths expected by",
                "`results/aaai_stress/oral_rerun_command_manifest.csv`.",
                "",
                "Data is intentionally not included in the public handoff package.",
                "",
            ]
        )
    )

    (out_dir / ".gitignore").write_text(
        "\n".join(
            [
                "__pycache__/",
                "*.py[cod]",
                ".venv*/",
                ".pytest_cache/",
                ".mypy_cache/",
                ".ruff_cache/",
                ".matplotlib_cache/",
                "data/gift-eval/",
                "external/",
                "",
            ]
        )
    )

    print(f"Prepared remote handoff repo at {out_dir}")
    print("Next local steps after GitHub auth is restored:")
    print(f"  cd {out_dir}")
    print("  git init")
    print("  git add .")
    print('  git commit -m "Prepare remote q9 rerun handoff"')
    print("  gh repo create lio-snp/tsfm-drcr-remote-q9 --public --source . --remote origin --push")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-clean", action="store_true", help="Do not remove an existing staging directory first.")
    args = parser.parse_args()
    prepare(args.out_dir.resolve(), clean=not args.no_clean)


if __name__ == "__main__":
    main()
