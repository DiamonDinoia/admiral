# Run argv[1:] in a job object. Append one line to $ADM_PROBE_LOG:
# commit_job_KiB, commit_max_KiB, ws_max_KiB, ws_child_KiB, seconds, max_image, object.
# commit_job_KiB: PeakProcessMemoryUsed of the job (largest peak commit of one process).
# commit_max_KiB, ws_max_KiB: largest PeakPagefileUsage and PeakWorkingSetSize over the
# job's processes. cl.exe is the driver and c1xx.exe does the compile, so both are sampled.
# ws_child_KiB: PeakWorkingSetSize of the process the launcher started (cl.exe).
import ctypes, os, subprocess, sys, time
from ctypes import wintypes as w

argv = sys.argv[1:]
t0 = time.time()
k = ctypes.WinDLL("kernel32", use_last_error=True)
k.CreateJobObjectW.restype = w.HANDLE
k.GetCurrentProcess.restype = w.HANDLE
k.OpenProcess.restype = w.HANDLE
k.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
k.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
k.QueryInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD,
                                        ctypes.c_void_p]
k.K32GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.c_void_p, w.DWORD]
k.QueryFullProcessImageNameW.argtypes = [w.HANDLE, w.DWORD, w.LPWSTR, ctypes.POINTER(w.DWORD)]


class IO(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in "r w o rb wb ob".split()]


class BASIC(ctypes.Structure):
    _fields_ = [("pu", ctypes.c_longlong), ("ju", ctypes.c_longlong), ("flags", w.DWORD),
                ("minws", ctypes.c_size_t), ("maxws", ctypes.c_size_t), ("active", w.DWORD),
                ("aff", ctypes.c_size_t), ("prio", w.DWORD), ("sched", w.DWORD)]


class EXT(ctypes.Structure):
    _fields_ = [("basic", BASIC), ("io", IO), ("pml", ctypes.c_size_t),
                ("jml", ctypes.c_size_t), ("peak_proc", ctypes.c_size_t),
                ("peak_job", ctypes.c_size_t)]


class PIDS(ctypes.Structure):
    _fields_ = [("assigned", w.DWORD), ("listed", w.DWORD),
                ("ids", ctypes.c_size_t * 256)]


class PMC(ctypes.Structure):
    _fields_ = [("cb", w.DWORD), ("faults", w.DWORD)] + [
        (n, ctypes.c_size_t) for n in "peak_ws ws qpp qp qpn qn commit peak_commit".split()]


def counters(h):
    c = PMC()
    c.cb = ctypes.sizeof(c)
    if not k.K32GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
        sys.exit("launcher: GetProcessMemoryInfo failed, error %d" % ctypes.get_last_error())
    return c.peak_ws // 1024, c.peak_commit // 1024


def image(h):
    buf = ctypes.create_unicode_buffer(1024)
    n = w.DWORD(1024)
    return os.path.basename(buf.value) if k.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)) else "?"


job = k.CreateJobObjectW(None, None)
if not job or not k.AssignProcessToJobObject(job, k.GetCurrentProcess()):
    sys.exit("launcher: job object setup failed, error %d" % ctypes.get_last_error())
QUERY = 0x0410  # PROCESS_QUERY_INFORMATION | PROCESS_VM_READ
p = subprocess.Popen(argv)
child = k.OpenProcess(QUERY, False, p.pid)
if not child:
    sys.exit("launcher: OpenProcess on the child failed")
seen = {p.pid: child}
while True:
    try:
        rc = p.wait(timeout=0.05)
        break
    except subprocess.TimeoutExpired:
        pass
    ids = PIDS()
    if not k.QueryInformationJobObject(job, 3, ctypes.byref(ids), ctypes.sizeof(ids), None):
        sys.exit("launcher: job pid list failed")
    for pid in ids.ids[:ids.listed]:
        if pid != os.getpid() and pid not in seen:
            h = k.OpenProcess(QUERY, False, pid)
            if h:
                seen[pid] = h
info = EXT()
if not k.QueryInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info), None):
    sys.exit("launcher: job query failed")
rows = [(counters(h), image(h)) for h in seen.values()]
ws_max, commit_max = max(r[0][0] for r in rows), max(r[0][1] for r in rows)
top = max(rows, key=lambda r: r[0][1])[1]
obj = argv[-1]
for i, a in enumerate(argv):
    if a == "-o" and i + 1 < len(argv):
        obj = argv[i + 1]
    elif a[:3] in ("/Fo", "-Fo") and len(a) > 3:
        obj = a[3:]
with open(os.environ["ADM_PROBE_LOG"], "a") as f:
    f.write("%d\t%d\t%d\t%d\t%.1f\t%s\t%s\n" % (
        info.peak_proc // 1024, commit_max, ws_max, counters(child)[0], time.time() - t0, top, obj))
sys.exit(rc)
