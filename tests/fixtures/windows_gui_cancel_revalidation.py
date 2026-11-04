from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

os.environ["QT_QPA_PLATFORM"] = "windows"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from video_heatmap.tool_app.config import Method
from video_heatmap.tool_app.main_window import AppState, MainWindow


ATTEMPT = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
OUT = ROOT / "outputs" / "U1-B4R" / "real_windows_gui_cancel" / ATTEMPT
OUT.mkdir(parents=True, exist_ok=True)


class ProcessEntry32W(ctypes.Structure):
    _fields_ = (
        ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260),
    )


def process_table() -> dict[int, dict[str, object]]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = (wintypes.HANDLE, ctypes.POINTER(ProcessEntry32W))
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = (wintypes.HANDLE, ctypes.POINTER(ProcessEntry32W))
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    if snapshot == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    entry = ProcessEntry32W()
    entry.dwSize = ctypes.sizeof(entry)
    result: dict[int, dict[str, object]] = {}
    try:
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            result[int(entry.th32ProcessID)] = {
                "pid": int(entry.th32ProcessID),
                "parent_pid": int(entry.th32ParentProcessID),
                "image": entry.szExeFile,
            }
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return result


def tree_pids(root_pid: int, table: dict[int, dict[str, object]]) -> list[int]:
    found = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, item in table.items():
            if int(item["parent_pid"]) in found and pid not in found:
                found.add(pid)
                changed = True
    return sorted(found)


def process_details(pid: int, table: dict[int, dict[str, object]]) -> dict[str, object]:
    base = dict(table.get(pid, {"pid": pid, "exists": False}))
    if pid not in table:
        return base
    command = (
        f"$p=Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\"; "
        "$o=Invoke-CimMethod -InputObject $p -MethodName GetOwner; "
        "[pscustomobject]@{command_line=$p.CommandLine;executable_path=$p.ExecutablePath;"
        "owner=($o.Domain+'\\'+$o.User)}|ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode == 0 and completed.stdout.strip():
        details = json.loads(completed.stdout)
        if details.get("owner") == "\\":
            details["owner"] = None
        base.update(details)
        if not details.get("command_line"):
            base["detail_error"] = "CIM process details were not accessible to this non-elevated session."
    else:
        base["detail_error"] = completed.stderr.strip()
    base["exists"] = True
    return base


def token_is_elevated() -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = ()
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    advapi32.OpenProcessToken.argtypes = (
        wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE),
    )
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = (
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    )
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(
        kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token),
    ):
        raise ctypes.WinError()
    elevation = wintypes.DWORD()
    size = wintypes.DWORD()
    try:
        if not advapi32.GetTokenInformation(
            token, 20, ctypes.byref(elevation), ctypes.sizeof(elevation), ctypes.byref(size),
        ):
            raise ctypes.WinError()
        return bool(elevation.value)
    finally:
        kernel32.CloseHandle(token)


def pump_until(app: QApplication, predicate, timeout_seconds: float, label: str) -> None:
    deadline = monotonic() + timeout_seconds
    while not predicate() and monotonic() < deadline:
        app.processEvents()
        QTest.qWait(20)
    if not predicate():
        raise TimeoutError(f"Timed out waiting for {label}")


def main() -> int:
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    participant = ROOT / "data" / "Face-to-Face" / "P5"
    gaze = next(participant.glob("*CsvData.txt"))
    video = next(path for path in participant.glob("*Scene Cam.mp4") if "Original" not in path.name)
    window.gaze_picker.edit.setText(str(gaze))
    window.video_picker.edit.setText(str(video))
    window.output_edit.setText(str(OUT / "runs"))
    window.method_combo.setCurrentIndex(window.method_combo.findData(Method.FLOW))
    window.end_frame.setValue(300)
    window.update_interval.setValue(1)
    window._input_paths_changed()
    window.end_frame.setValue(300)
    window._refresh_validation()
    started = monotonic()
    window.generate_button.click()
    pump_until(app, lambda: window.runner.root_pid > 0, 15, "QProcess root PID")
    root_pid = window.runner.root_pid
    pump_until(
        app,
        lambda: len(tree_pids(root_pid, process_table())) >= 2,
        20,
        "delegated backend descendant",
    )
    QTest.qWait(1000)
    app.processEvents()
    before_table = process_table()
    before_pids = tree_pids(root_pid, before_table)
    before = [process_details(pid, before_table) for pid in before_pids]
    sentinel_before = process_details(sentinel.pid, before_table)
    window.grab().save(str(OUT / "gui_before_cancel.png"))
    before_time = datetime.now(timezone.utc).isoformat()

    cancel_started = monotonic()
    window.cancel_button.click()
    pump_until(app, lambda: window.state is not AppState.PROCESSING, 15, "GUI terminal state")
    cancel_to_terminal_seconds = monotonic() - cancel_started
    pump_until(app, lambda: not window.runner.running, 15, "QProcess exit")
    QTest.qWait(300)
    app.processEvents()
    after_table = process_table()
    after = [process_details(pid, after_table) for pid in before_pids]
    sentinel_after = process_details(sentinel.pid, after_table)
    window.grab().save(str(OUT / "gui_after_cancel.png"))
    summary_path = window.current_config.output_dir / "tool_run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evidence = {
        "schema_version": "u1_b4r_windows_gui_cancellation.v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "platform": sys.platform,
        "qt_platform": QApplication.platformName(),
        "current_process_pid": os.getpid(),
        "current_process_elevated": token_is_elevated(),
        "current_user": subprocess.run(
            ["whoami"], capture_output=True, text=True, check=False,
        ).stdout.strip(),
        "owner_basis": "QProcess root and descendants inherit the non-elevated GUI user's token; CIM command-line/owner reads were access-limited.",
        "mechanism": "Windows Job Object assigned to the QProcess root before delegated child startup",
        "cancel_requested_at": before_time,
        "generate_to_terminal_seconds": monotonic() - started,
        "cancel_to_terminal_seconds": cancel_to_terminal_seconds,
        "root_pid": root_pid,
        "pid_tree_before": before,
        "pid_tree_after": after,
        "all_recorded_run_pids_ended": all(not item.get("exists", False) for item in after),
        "unrelated_python_before": sentinel_before,
        "unrelated_python_after": sentinel_after,
        "unrelated_python_survived": sentinel.poll() is None,
        "gui_state": window.state.value,
        "preview_enabled": window.preview_button.isEnabled(),
        "screenshot_enabled": window.screenshot_button.isEnabled(),
        "export_enabled": window.export_button.isEnabled(),
        "summary_path": str(summary_path),
        "summary": summary,
        "stdout_stderr_log_path": str(OUT / "gui_stdout_stderr.log"),
        "screenshots": [str(OUT / "gui_before_cancel.png"), str(OUT / "gui_after_cancel.png")],
    }
    (OUT / "pid_tree_before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
    (OUT / "pid_tree_after.json").write_text(json.dumps(after, indent=2), encoding="utf-8")
    (OUT / "gui_stdout_stderr.log").write_text(window.log_view.toPlainText(), encoding="utf-8")
    (OUT / "revalidation_evidence.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps({
        "gui_state": evidence["gui_state"],
        "all_recorded_run_pids_ended": evidence["all_recorded_run_pids_ended"],
        "unrelated_python_survived": evidence["unrelated_python_survived"],
        "summary_status": summary["status"],
        "exportable": summary["exportable"],
        "elevated": evidence["current_process_elevated"],
        "evidence": str(OUT / "revalidation_evidence.json"),
    }, indent=2), flush=True)
    sentinel.terminate()
    sentinel.wait(timeout=5)
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
