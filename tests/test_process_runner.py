import os
from pathlib import Path
import tempfile
import time
import unittest

from PySide6.QtCore import QCoreApplication

from pipeline_manager import PipelineWorker


class ProcessRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def test_openmvs_stdout_is_drained_while_a_stale_log_exists(self):
        """A full stdout pipe must not freeze TextureMesh log tailing."""
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            executable = work_dir / "TextureMesh"
            executable.write_text(
                "#!/bin/sh\n"
                "i=0\n"
                "while [ $i -lt 12000 ]; do\n"
                "  echo 'TextureMesh output that must be drained from stdout'\n"
                "  i=$((i + 1))\n"
                "done\n"
            )
            executable.chmod(0o755)
            (work_dir / "TextureMesh-stale.log").write_text("old completed run\n")

            worker = PipelineWorker(None, directory)
            started = time.monotonic()
            ok = worker._run_process_realtime(
                [str(executable)], timeout=8.0, cwd=directory, env=os.environ.copy()
            )

            self.assertTrue(ok)
            self.assertLess(time.monotonic() - started, 7.0)
            self.assertIn(
                "TextureMesh output that must be drained from stdout",
                worker.last_output_lines,
            )


if __name__ == "__main__":
    unittest.main()
