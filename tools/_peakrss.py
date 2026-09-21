"""Peak resident memory of the current process, in MiB.

One implementation, shared by both workers. The Windows path needs explicit
`argtypes`/`restype`: without them ctypes truncates the process HANDLE to 32
bits, `GetProcessMemoryInfo` returns 0, and the caller reports a peak of 0.0 MiB
that looks like a measurement and is not one. That silent zero is the reason
this lives in its own file with its own check.
"""
from __future__ import annotations

import os


def peak_rss_mib() -> float:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                               ctypes.POINTER(_Counters),
                                               wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        c = _Counters()
        c.cb = ctypes.sizeof(c)
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                          ctypes.byref(c), c.cb):
            return float("nan")     # unknown, and says so
        return c.PeakWorkingSetSize / 2 ** 20

    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


if __name__ == "__main__":
    print(f"{peak_rss_mib():.1f} MiB")
