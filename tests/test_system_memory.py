import unittest
from types import SimpleNamespace

import context  # noqa: F401
from low_snr_tsfm.system_memory import (
    parse_vm_stat_available_gb,
    windows_available_ram_gb,
    windows_commit_fraction,
)


class SystemMemoryTests(unittest.TestCase):
    def test_parse_vm_stat_uses_reported_page_size(self):
        output = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                4000.
Pages active:                             67000.
Pages inactive:                           60000.
Pages speculative:                         6000.
Pages wired down:                        151000.
"""
        expected = (4000 + 60000 + 6000) * 16384 / (1024**3)
        self.assertAlmostEqual(parse_vm_stat_available_gb(output), expected)

    def test_parse_vm_stat_defaults_to_4k_pages(self):
        output = """Mach Virtual Memory Statistics:
Pages free:                                1024.
Pages inactive:                           1024.
Pages speculative:                         0.
"""
        expected = 2048 * 4096 / (1024**3)
        self.assertAlmostEqual(parse_vm_stat_available_gb(output), expected)

    def test_windows_memory_conversions(self):
        gib = 1024**3
        status = SimpleNamespace(
            ullAvailPhys=18 * gib,
            ullTotalPageFile=32 * gib,
            ullAvailPageFile=24 * gib,
        )
        self.assertEqual(windows_available_ram_gb(status), 18.0)
        self.assertEqual(windows_commit_fraction(status), 0.25)

    def test_windows_commit_fraction_handles_missing_limit(self):
        status = SimpleNamespace(ullTotalPageFile=0, ullAvailPageFile=0)
        self.assertIsNone(windows_commit_fraction(status))


if __name__ == "__main__":
    unittest.main()
