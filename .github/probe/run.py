# Usage: run.py <compile_commands.json> <log> <variant>...
# A variant is name:DEFINE=V,DEFINE=V. Compile the probe copy of test_ct_root.cpp with each
# set of defines under the launcher, in the compile command CMake wrote for the real TU.
import json, os, subprocess, sys

cc, log = sys.argv[1], sys.argv[2]
launcher = os.path.join(os.path.dirname(os.path.abspath(__file__)), "launcher.py")
probe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_ct_root.cpp")
ents = [e for e in json.load(open(cc)) if e["file"].replace("\\", "/").endswith("kernels/test_ct_root.cpp")
        and "poison" not in e["output"]]
if len(ents) != 1:
    sys.exit("run: expected one test_ct_root entry, got %d" % len(ents))
e = ents[0]
if e["file"] not in e["command"] or "test_ct_root.cpp.obj" not in e["command"]:
    sys.exit("run: source or object not in the compile command")
env = dict(os.environ, ADM_PROBE_LOG=log)
status = 0
for v in sys.argv[3:]:
    name, _, defs = v.partition(":")
    cmd = e["command"].replace(e["file"], probe).replace("test_ct_root.cpp.obj", "probe_%s.obj" % name)
    cmd += "".join(" /D%s" % d for d in defs.split(",") if d)
    cmd = '"%s" "%s" %s' % (sys.executable, launcher, cmd)
    print("VARIANT", name, defs, flush=True)
    env["ADM_PROBE_NAME"] = name
    rc = subprocess.call(cmd, cwd=e["directory"], env=env)
    with open(log, "a") as f:
        f.write("# %s rc=%d\n" % (name, rc))
    status = status or rc
sys.exit(status)
