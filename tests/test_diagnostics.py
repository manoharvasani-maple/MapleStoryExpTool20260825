import tempfile
from pathlib import Path
import unittest

from diagnostics import configure_diagnostics, get_logger, shutdown_diagnostics


class DiagnosticsTests(unittest.TestCase):
    def tearDown(self):
        shutdown_diagnostics()

    def test_writes_utf8_diagnostic_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "error.log"
            configured_path = configure_diagnostics(
                "test",
                log_path=log_path,
                install_hooks=False,
            )

            get_logger("test").error("測試擷取錯誤")
            shutdown_diagnostics()

            self.assertEqual(configured_path, log_path)
            content = log_path.read_text(encoding="utf-8")
            self.assertIn("Application started version=test", content)
            self.assertIn("測試擷取錯誤", content)

    def test_rotates_log_instead_of_growing_without_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "error.log"
            configure_diagnostics(
                "test",
                log_path=log_path,
                max_bytes=300,
                backup_count=2,
                install_hooks=False,
            )

            logger = get_logger("rotation")
            for index in range(20):
                logger.error("entry=%s payload=%s", index, "x" * 80)
            shutdown_diagnostics()

            self.assertTrue(log_path.exists())
            self.assertTrue(Path(f"{log_path}.1").exists())
            self.assertLessEqual(len(list(log_path.parent.glob("error.log*"))), 3)


if __name__ == "__main__":
    unittest.main()

