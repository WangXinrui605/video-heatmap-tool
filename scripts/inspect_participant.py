from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.gaze_loader import load_gaze_txt, summarize_gaze
from video_heatmap.video_io import get_video_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    participant_dir = Path(args.participant_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(participant_dir.glob("*.txt"))
    scene_videos = sorted([p for p in participant_dir.glob("*.mp4") if "Original_Eye Tracking" not in p.name])
    reference_videos = sorted([p for p in participant_dir.glob("*.mp4") if "Original_Eye Tracking" in p.name])

    if not txt_files:
        raise SystemExit(f"No .txt files found in {participant_dir}")

    video_info = None
    video_error = None
    if scene_videos:
        try:
            video_info = get_video_info(scene_videos[0]).__dict__
            video_info["path"] = str(video_info["path"])
        except Exception as exc:
            video_error = repr(exc)

    width = video_info["width"] if video_info else None
    height = video_info["height"] if video_info else None
    gaze = load_gaze_txt(txt_files[0], video_width=width, video_height=height)

    report = {
        "participant_dir": str(participant_dir),
        "gaze_file": str(txt_files[0]),
        "scene_video": str(scene_videos[0]) if scene_videos else None,
        "reference_video": str(reference_videos[0]) if reference_videos else None,
        "video_info": video_info,
        "video_error": video_error,
        "gaze_summary": summarize_gaze(gaze),
        "columns": list(gaze.dataframe.columns),
    }

    report_path = out_dir / "inspection_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
