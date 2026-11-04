from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.tool_app.cancellation import CancellationScope, process_creation_flags


def pid_exists(pid: int) -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True, text=True, check=False,
    )
    return str(pid) in result.stdout


@unittest.skipUnless(os.name == "nt", "Windows process-tree contract")
class WindowsCancellationTests(unittest.TestCase):
    def test_cancel_targets_current_tree_and_preserves_unrelated_python(self) -> None:
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        root = None
        try:
            with tempfile.TemporaryDirectory() as name:
                pid_file = Path(name) / "pids.json"
                fixture = Path(__file__).parent / "fixtures" / "process_tree_root.py"
                scope = CancellationScope()
                root = subprocess.Popen(
                    [sys.executable, str(fixture), str(pid_file)],
                    creationflags=process_creation_flags(),
                )
                scope.attach(root.pid)
                deadline = time.monotonic() + 10
                while not pid_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                pids = json.loads(pid_file.read_text(encoding="utf-8"))
                result = scope.cancel()
                root.wait(timeout=10)
                deadline = time.monotonic() + 5
                while pid_exists(pids["child"]) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.mechanism, "windows-job-object")
                self.assertFalse(pid_exists(pids["child"]))
                self.assertTrue(pid_exists(unrelated.pid))
                scope.close()
        finally:
            if root is not None and root.poll() is None:
                root.kill()
            unrelated.kill()
            unrelated.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
