"""Small cross-platform memory helpers for local model preflights."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any


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


def _windows_memory_status() -> Any:
    """Return the native Windows memory status without optional dependencies."""
    import ctypes
    from ctypes import wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return status


def windows_available_ram_gb(status: Any | None = None) -> float:
    """Return available Windows physical RAM in GiB."""
    current = status if status is not None else _windows_memory_status()
    return float(current.ullAvailPhys) / (1024**3)


def windows_commit_fraction(status: Any | None = None) -> float | None:
    """Return Windows committed-memory use as a fraction of its limit."""
    current = status if status is not None else _windows_memory_status()
    total = float(current.ullTotalPageFile)
    if total <= 0:
        return None
    available = min(max(float(current.ullAvailPageFile), 0.0), total)
    return (total - available) / total


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
        if sys.platform == "win32":
            return windows_available_ram_gb()
    except Exception:  # noqa: BLE001 - preflight reports unknown as zero
        return 0.0
    return 0.0
