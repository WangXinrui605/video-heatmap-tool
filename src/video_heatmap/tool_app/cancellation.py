from __future__ import annotations

from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import os
import signal
import subprocess


WINDOWS_PROCESS_FLAGS = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


@dataclass(frozen=True)
class CancellationResult:
    root_pid: int
    returncode: int
    stdout: str
    stderr: str
    mechanism: str = "process-group"


def process_creation_flags() -> int:
    """Compatibility flags for subprocess fixtures on Windows."""
    return WINDOWS_PROCESS_FLAGS if os.name == "nt" else 0


if os.name == "nt":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    )
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    _kernel32.WaitForSingleObject.restype = wintypes.DWORD
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel32.CloseHandle.restype = wintypes.BOOL

    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _WAIT_OBJECT_0 = 0
    _WAIT_TIMEOUT = 258

    class _JobBasicLimitInformation(ctypes.Structure):
        _fields_ = (
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        )

    class _IoCounters(ctypes.Structure):
        _fields_ = tuple((name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        ))

    class _JobExtendedLimitInformation(ctypes.Structure):
        _fields_ = (
            ("BasicLimitInformation", _JobBasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        )


def _windows_error(action: str) -> OSError:
    code = ctypes.get_last_error()
    return OSError(code, f"{action}: {ctypes.FormatError(code).strip()}")


class CancellationScope:
    """Own one run tree for targeted, same-integrity cancellation.

    Windows descendants inherit membership after the QProcess root is assigned to
    this Job Object. No process-name enumeration or administrator privilege is used.
    """

    def __init__(self) -> None:
        self.root_pid = 0
        self._handle: int | None = None
        if os.name == "nt":
            handle = _kernel32.CreateJobObjectW(None, None)
            if not handle:
                raise _windows_error("CreateJobObjectW failed")
            information = _JobExtendedLimitInformation()
            information.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not _kernel32.SetInformationJobObject(
                handle,
                _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(information),
                ctypes.sizeof(information),
            ):
                error = _windows_error("SetInformationJobObject failed")
                _kernel32.CloseHandle(handle)
                raise error
            self._handle = handle

    def attach(self, root_pid: int) -> None:
        if root_pid <= 0 or root_pid == os.getpid():
            raise ValueError("Refusing an invalid or current-process root PID.")
        self.root_pid = root_pid
        if os.name != "nt":
            return
        access = _PROCESS_TERMINATE | _PROCESS_SET_QUOTA | _PROCESS_QUERY_LIMITED_INFORMATION
        process = _kernel32.OpenProcess(access, False, root_pid)
        if not process:
            raise _windows_error(f"OpenProcess({root_pid}) failed")
        try:
            if not _kernel32.AssignProcessToJobObject(self._handle, process):
                raise _windows_error(f"AssignProcessToJobObject({root_pid}) failed")
        finally:
            _kernel32.CloseHandle(process)

    def cancel(self, *, timeout_seconds: float = 10.0) -> CancellationResult:
        if self.root_pid <= 0:
            raise ValueError("Cancellation scope has no attached root PID.")
        if os.name != "nt":
            try:
                os.killpg(self.root_pid, signal.SIGTERM)
            except ProcessLookupError:
                return CancellationResult(
                    self.root_pid, 128, "", "process tree already exited", "process-group",
                )
            return CancellationResult(
                self.root_pid, 0, "sent SIGTERM to process group", "", "process-group",
            )

        if not self._handle:
            return CancellationResult(
                self.root_pid, 6, "", "Windows Job Object handle is closed.", "windows-job-object",
            )
        if not _kernel32.TerminateJobObject(self._handle, 0xC000013A):
            error = _windows_error("TerminateJobObject failed")
            return CancellationResult(
                self.root_pid, int(error.errno or 1), "", str(error), "windows-job-object",
            )
        timeout_ms = max(0, min(int(timeout_seconds * 1000), 0xFFFFFFFE))
        wait_result = _kernel32.WaitForSingleObject(self._handle, timeout_ms)
        if wait_result == _WAIT_OBJECT_0:
            return CancellationResult(
                self.root_pid, 0, "Current run Job Object terminated; all members exited.", "",
                "windows-job-object",
            )
        if wait_result == _WAIT_TIMEOUT:
            return CancellationResult(
                self.root_pid, 1460, "",
                f"Timed out after {timeout_seconds:g}s waiting for the current run tree to exit.",
                "windows-job-object",
            )
        error = _windows_error("WaitForSingleObject(job) failed")
        return CancellationResult(
            self.root_pid, int(error.errno or 1), "", str(error), "windows-job-object",
        )

    def close(self) -> None:
        if os.name == "nt" and self._handle:
            _kernel32.CloseHandle(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()


def cancel_process_tree(root_pid: int, *, timeout_seconds: float = 10.0) -> CancellationResult:
    """Cancel a POSIX process group.

    Windows requires a ``CancellationScope`` created before QProcess startup and
    attached immediately afterward. Attaching only when Cancel is clicked cannot
    safely include children that already exist.
    """
    if os.name == "nt":
        return CancellationResult(
            root_pid, 50, "", "Windows cancellation requires a pre-attached Job Object scope.",
            "windows-job-object",
        )
    if root_pid <= 0 or root_pid == os.getpid():
        raise ValueError("Refusing an invalid or current-process root PID.")
    try:
        os.killpg(root_pid, signal.SIGTERM)
    except ProcessLookupError:
        return CancellationResult(root_pid, 128, "", "process tree already exited")
    return CancellationResult(root_pid, 0, "sent SIGTERM to process group", "")
