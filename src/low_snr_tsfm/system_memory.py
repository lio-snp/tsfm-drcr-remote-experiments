"""Small cross-platform memory helpers for local model preflights."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


DARWIN_AVAILABLE_PAGE_LABELS = {
    "Pages free",
    "Pages inactive",
    "Pages speculative",
}


def parse_vm_stat_available_gb(output: str) -> float:
    """Estimate reclaimable local RAM from macOS ``vm_stat`` output."""
    page_size = 4096
    page_size_match = re.search(r"page size of (\d+) bytes", output)
    if page_size_match:
        page_size = int(page_size_match.group(1))

    pages = 0
    for line in output.splitlines():
        label, separator, raw_value = line.partition(":")
        if not separator or label.strip() not in DARWIN_AVAILABLE_PAGE_LABELS:
            continue
        value_match = re.search(r"\d+", raw_value.replace(".", ""))
        if value_match:
            pages += int(value_match.group(0))
    return pages * page_size / (1024**3)


def parse_meminfo_available_gb(path: Path = Path("/proc/meminfo")) -> float:
    with path.open() as handle:
        for line in handle:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def available_ram_gb() -> float:
    try:
        if sys.platform == "darwin":
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True,
                text=True,
                check=True,
            )
            return parse_vm_stat_available_gb(result.stdout)
        if sys.platform == "linux":
            return parse_meminfo_available_gb()
    except Exception:  # noqa: BLE001 - preflight reports unknown as zero
        return 0.0
    return 0.0
