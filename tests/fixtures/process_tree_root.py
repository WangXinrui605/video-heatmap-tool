from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time


pid_file = Path(sys.argv[1])
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
pid_file.write_text(json.dumps({"root": __import__("os").getpid(), "child": child.pid}), encoding="utf-8")
while True:
    time.sleep(1)
