import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_benefit_selective_external_queue import ROOT, split_job_command, status_is_ok


class BenefitSelectiveExternalQueueTests(unittest.TestCase):
    def test_status_is_ok_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            self.assertFalse(status_is_ok(path))

            path.write_text("{broken", encoding="utf-8")
            self.assertFalse(status_is_ok(path))

            path.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
            self.assertFalse(status_is_ok(path))

            path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
            self.assertTrue(status_is_ok(path))

    def test_split_job_command_resolves_windows_venv_python(self) -> None:
        with patch("scripts.run_benefit_selective_external_queue.sys.platform", "win32"):
            command = split_job_command(
                ".venv-chronos/bin/python scripts/run.py --dataset-name m4_yearly"
            )
        self.assertEqual(command[0], str(ROOT / ".venv-chronos" / "Scripts" / "python.exe"))
        self.assertEqual(command[1:], ["scripts/run.py", "--dataset-name", "m4_yearly"])

    def test_split_job_command_is_unchanged_on_posix(self) -> None:
        with patch("scripts.run_benefit_selective_external_queue.sys.platform", "linux"):
            command = split_job_command(".venv-moirai/bin/python scripts/run.py")
        self.assertEqual(command, [".venv-moirai/bin/python", "scripts/run.py"])


if __name__ == "__main__":
    unittest.main()
