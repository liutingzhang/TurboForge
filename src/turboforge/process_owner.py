"""Own Windows model servers so an abruptly closed UI cannot orphan them."""
import os
import ctypes
from ctypes import wintypes


class ServerJob:
    def __init__(self):
        self.handle = None
        if os.name != "nt":
            return
        class Basic(ctypes.Structure):
            _fields_ = [("process_time",ctypes.c_int64),("job_time",ctypes.c_int64),("flags",wintypes.DWORD),
                        ("minimum",ctypes.c_size_t),("maximum",ctypes.c_size_t),("active",wintypes.DWORD),
                        ("affinity",ctypes.c_size_t),("priority",wintypes.DWORD),("scheduling",wintypes.DWORD)]
        class Extended(ctypes.Structure):
            _fields_ = [("basic",Basic),("io",ctypes.c_uint64*6),("process_memory",ctypes.c_size_t),
                        ("job_memory",ctypes.c_size_t),("peak_process",ctypes.c_size_t),("peak_job",ctypes.c_size_t)]
        kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype=wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        self.kernel=kernel
        handle=kernel.CreateJobObjectW(None,None)
        if not handle: raise ctypes.WinError(ctypes.get_last_error())
        info=Extended();info.basic.flags=0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(handle,9,ctypes.byref(info),ctypes.sizeof(info)):
            error=ctypes.get_last_error();kernel.CloseHandle(handle);raise ctypes.WinError(error)
        self.handle=handle

    def attach(self, pid):
        if self.handle is None:return
        process=self.kernel.OpenProcess(0x0101,False,pid)  # SET_QUOTA | TERMINATE
        if not process:raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self.kernel.AssignProcessToJobObject(self.handle,process):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:self.kernel.CloseHandle(process)

    def close(self):
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle);self.handle=None

    def __del__(self):
        self.close()
