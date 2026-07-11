import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_benefit_selective_external_queue import status_is_ok


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


if __name__ == "__main__":
    unittest.main()
