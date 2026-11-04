from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        from video_heatmap.tool_app.main_window import MainWindow
    except ImportError as exc:
        print(
            "Video Heatmap Tool could not start because PySide6 6.11.1 or another project "
            f"dependency is unavailable in {sys.executable}.\n{exc}",
            file=sys.stderr,
        )
        return 2
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Video Heatmap Tool")
    try:
        window = MainWindow()
        window.show()
    except Exception as exc:
        QMessageBox.critical(None, "Video Heatmap Tool startup failed", str(exc))
        return 1
    if "--smoke-test" in sys.argv:
        app.processEvents()
        print(f"GUI_SMOKE_OK state={window.state.value}")
        window.close()
        return 0
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
