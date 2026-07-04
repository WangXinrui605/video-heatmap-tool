from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_dataset(data_root: Path) -> dict:
    participants = []
    for participant_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        txt_files = sorted(participant_dir.glob("*.txt"))
        mp4_files = sorted(participant_dir.glob("*.mp4"))
        participants.append(
            {
                "participant": participant_dir.name,
                "txt_files": [str(p) for p in txt_files],
                "mp4_files": [str(p) for p in mp4_files],
            }
        )
    return {"data_root": str(data_root), "participants": participants}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Path to Face-to-Face dataset folder")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"Data root does not exist: {data_root}")
    print(json.dumps(inspect_dataset(data_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
