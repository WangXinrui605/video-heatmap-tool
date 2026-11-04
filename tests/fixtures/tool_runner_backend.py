from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from video_heatmap.tool_app.artifacts import RunStatus, new_run_record, transition_run, write_json_atomic
from video_heatmap.tool_app.config import Method


status = RunStatus(sys.argv[1])
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
record = new_run_record(Method.BASELINE, {}, output, run_id="fixture-run")
write_json_atomic(output / "tool_run_summary.json", record)
print("fixture stdout", flush=True)
print("fixture stderr", file=sys.stderr, flush=True)
if status is RunStatus.COMPLETED:
    primary = output / "dynamic" / "heatmap_development.mp4"
    primary.parent.mkdir(parents=True, exist_ok=True)
    primary.touch()
record = transition_run(record, status, output, error="fixture failure" if status is RunStatus.FAILED else None)
write_json_atomic(output / "tool_run_summary.json", record)
