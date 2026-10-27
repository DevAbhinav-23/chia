"""Unit tests for circt_exec_probe, circt_reduce_run and circt_symbolize.

No CIRCT build, Docker image or cluster needed: every test drives the three
functions with ordinary host programs (/bin/sh, /bin/true, this Python) or a
stub in tmp_path, and calls each node's undecorated function, so no Ray starts.
Needs `prlimit` (util-linux) on PATH.
Run: pytest chia/chipyard/test/test_circt_probe.py
"""

import ast
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from chia.chipyard import circt
from chia.chipyard.circt import (CPU_HARD_MARGIN_SECONDS, circt_exec_probe,
                                 circt_reduce_run, circt_symbolize)

_LIMITS = dict(wall_seconds=30, address_space_bytes=4 * 1024 ** 3, cpu_seconds=20)

#: The functions this file covers, and the private helpers they call.
_FUNCTIONS = ("circt_exec_probe", "circt_reduce_run", "circt_symbolize",
              "_symbolize_module", "_in_circt_object", "allocation_evidence",
              "_limit_hit", "_drain", "_reap", "_signal_name", "_killpg",
              "_node_id")

pytestmark = pytest.mark.skipif(shutil.which("prlimit") is None,
                                reason="prlimit (util-linux) not on PATH")


def call_node(fn, *args, **kwargs):
    """Call a @ChiaFunction node's undecorated function, with no Ray."""
    return getattr(fn, "_chia_original", fn)(*args, **kwargs)


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(Path(circt.__file__).read_text())
    node, = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name == name]
    return node


def _code_of(name: str) -> str:
    """The source of one function of circt.py, its docstring removed."""
    node = _function(name)
    body = node.body[1:] if ast.get_docstring(node) else node.body
    return "\n".join(ast.unparse(stmt) for stmt in body)


def _ray_session() -> bool:
    """Whether this process is already inside a Ray session, without starting one."""
    ray = sys.modules.get("ray")
    return bool(ray is not None and ray.is_initialized())


def _run(tool: str, argv: list, tmp_path: Path, **over):
    """One circt_exec_probe call with small limits."""
    return call_node(circt_exec_probe, tool, argv, cwd=str(tmp_path),
                     **{**_LIMITS, **over})


def _stub(tmp_path: Path, name: str, body: str) -> str:
    """Write an executable /bin/sh stub and return its path."""
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + textwrap.dedent(body))
    path.chmod(0o755)
    return str(path)


# --- circt_exec_probe -------------------------------------------------------

def test_the_prlimit_prefix(tmp_path):
    """The argv, in order, long spellings, and no --rss."""
    out = _run("/bin/true", ["--x"], tmp_path, address_space_bytes=1073741824,
               cpu_seconds=5, nofile=256)
    assert out["argv"] == ["prlimit", "--as=1073741824", "--cpu=5:10",
                           "--nofile=256", "--", "/bin/true", "--x"]
    assert CPU_HARD_MARGIN_SECONDS == 5
    assert "--rss" not in " ".join(out["argv"])


def test_all_three_limits_reach_the_child(tmp_path):
    out = _run("/bin/sh", ["-c", "ulimit -v; ulimit -t; ulimit -n"], tmp_path,
               address_space_bytes=1073741824, cpu_seconds=5, nofile=256)
    assert out["stdout"].split() == ["1048576", "5", "256"]
    assert out["exit_status"] == 0 and out["signal"] is None


def test_the_whole_process_group_dies_on_timeout(tmp_path):
    out = _run("/bin/sh", ["-c", "sleep 30 & echo $!; sleep 30"], tmp_path,
               wall_seconds=1)
    grandchild = int(out["stdout"].split()[0])
    assert out["limit_hit"] == "wall"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    pytest.fail(f"grandchild {grandchild} outlived its process group")


def test_the_signal_is_a_field_of_its_own(tmp_path):
    """SIGABRT gives a null exit status, never 134."""
    out = _run(sys.executable,
               ["-c", "import os, signal; os.kill(os.getpid(), signal.SIGABRT)"],
               tmp_path)
    assert out["signal"] == "SIGABRT"
    assert out["exit_status"] is None
    code = _code_of("circt_exec_probe")
    for needed in ("os.fork()", "os.setsid()", "os.execvp(", "os.wait4("):
        assert needed in code + _code_of("_reap"), needed
    assert "subprocess.Popen" not in code


def test_output_is_capped_per_call(tmp_path):
    argv = ["-c", "print('a' * 100000)"]
    capped = _run(sys.executable, argv, tmp_path, output_byte_cap=1000)
    assert len(capped["stdout"]) == 1000 and capped["truncated"] is True
    whole = _run(sys.executable, argv, tmp_path, output_byte_cap=1_000_000)
    assert len(whole["stdout"]) == 100001 and whole["truncated"] is False


def test_missing_tool_raises_and_nothing_else_does(tmp_path):
    with pytest.raises(FileNotFoundError):
        _run(str(tmp_path / "no-such-tool"), [], tmp_path)
    # 126: prlimit starts, then fails to exec a file it may not run.
    not_executable = tmp_path / "plain.txt"
    not_executable.write_text("not a program\n")
    assert _run(str(not_executable), [], tmp_path)["exit_status"] == 126


def test_no_shell_and_no_command_string():
    for name in _FUNCTIONS:
        for node in ast.walk(_function(name)):
            if not isinstance(node, ast.Call):
                continue
            assert not any(kw.arg == "shell" for kw in node.keywords), name
            if getattr(node.func, "attr", "") in ("run", "Popen", "call",
                                                  "check_output"):
                assert not isinstance(node.args[0], ast.Constant), name


def test_wait4_supplies_cpu_and_peak_rss(tmp_path):
    out = _run(sys.executable, ["-c", "b = bytearray(300 * 1024 * 1024); b[0] = 1"],
               tmp_path)
    assert 200 * 1024 ** 2 < out["peak_rss_bytes"] < 700 * 1024 ** 2
    busy = _run(sys.executable,
                ["-c", "import time\nt=time.time()\nwhile time.time()-t<0.4: pass"],
                tmp_path)
    assert busy["cpu_seconds"] >= 0.2


def test_limit_hit_against_measured_prlimit(tmp_path):
    spin = _run(sys.executable, ["-c", "while True: pass"], tmp_path,
                cpu_seconds=1, wall_seconds=60)
    assert spin["signal"] == "SIGXCPU" and spin["limit_hit"] == "cpu"
    assert spin["exit_status"] is None

    # The control: prlimit with no hard margin kills with SIGKILL instead.
    control = subprocess.run(["prlimit", "--cpu=1", "--", sys.executable,
                              "-c", "while True: pass"])
    assert control.returncode == -signal.SIGKILL

    hog = _run(sys.executable, ["-c", "b = bytearray(500 * 1024 * 1024)"],
               tmp_path, address_space_bytes=104857600)
    assert hog["signal"] is None and hog["exit_status"] == 1
    assert "MemoryError" in hog["stderr"]
    assert hog["limit_hit"] == "address_space"

    assert _run("/bin/true", [], tmp_path)["limit_hit"] is None


def test_the_worker_identity_fields(tmp_path):
    before = _ray_session()
    out = _run("/bin/true", [], tmp_path)
    assert out["worker_hostname"]
    assert out["child_pid"] > 0
    # Empty exactly when there is no session to ask, and asking started none.
    assert (out["worker_node_id"] == "") is not before
    assert _ray_session() is before


# --- circt_reduce_run -------------------------------------------------------

def test_the_reducer_argv(monkeypatch):
    """--keep-best explicit, --test-must-fail never."""
    seen, signalled = [], []

    class _Popen:
        returncode = 0
        pid = os.getpid()                   # the reducer's `finally` signals this group

        def __init__(self, argv, **kwargs):
            seen.append(argv)
            assert kwargs["start_new_session"] is True

        def communicate(self, timeout=None):
            return (b"", None)

    monkeypatch.setattr(circt.subprocess, "Popen", _Popen)
    monkeypatch.setattr(circt, "_killpg", lambda pgid, *sig: signalled.append(pgid))
    call_node(circt_reduce_run, "in.mlir", "interesting.sh", "out.mlir",
              binary="/workspace/circt/build/bin/circt-reduce")
    assert signalled == [os.getpid()], "the stub is what keeps the suite alive here"
    assert seen == [["/workspace/circt/build/bin/circt-reduce", "in.mlir",
                     "--test=interesting.sh", "--keep-best", "-o", "out.mlir"]]
    assert "--test-must-fail" not in seen[0]


def test_output_valid_is_computed_after_the_exit(tmp_path):
    empty = tmp_path / "empty.mlir"
    empty.write_bytes(b"")
    out = call_node(circt_reduce_run, "in.mlir", "t.sh", str(empty),
                    binary="/bin/true")
    assert out["output_valid"] is False and out["success"] is True

    late = _stub(tmp_path, "late-reduce.sh", """
        sleep 0.3
        printf 'reduced\\n' > "$5"
    """)
    target = tmp_path / "reduced.mlir"
    out = call_node(circt_reduce_run, "in.mlir", "t.sh", str(target), binary=late)
    assert out["output_valid"] is True and out["wall_seconds"] >= 0.3


# --- circt_symbolize --------------------------------------------------------

def test_an_unparseable_line_is_unresolved(tmp_path):
    """Garbage on the symbolizer's stdout, or no symbolizer, yields a record."""
    frames = [{"index": 0, "address": "0x1", "shape": "module_offset",
               "module": "/lib/libX.so", "offset": "0x10",
               "function": "", "file": "", "line": 0}]
    garbage = _stub(tmp_path, "garbage.sh", """
        cat > /dev/null
        printf 'not json at all\\n'
    """)
    for symbolizer in (garbage, str(tmp_path / "absent")):
        out = call_node(circt_symbolize, frames, symbolizer=symbolizer)
        assert out[0]["function"] == "" and out[0]["file"] == ""
        assert out[0]["line"] == 0 and out[0]["in_circt_object"] is False


def test_one_symbolizer_call_per_module(tmp_path):
    """Frames are grouped by module and their offsets go on stdin."""
    log = tmp_path / "calls.log"
    fake = _stub(tmp_path, "sym.sh", f"""
        printf '%s\\n' "$*" >> {log}
        cat >> {log}
        printf -- '---\\n' >> {log}
    """)
    frames = [{"index": i, "address": "0x0", "shape": "module_offset",
               "module": module, "offset": offset,
               "function": "", "file": "", "line": 0}
              for i, (module, offset) in enumerate([("/lib/a.so", "0x10"),
                                                    ("/lib/b.so", "0x20"),
                                                    ("/lib/a.so", "0x30")])]
    call_node(circt_symbolize, frames, symbolizer=fake)
    calls = log.read_text().split("---\n")[:-1]
    assert len(calls) == 2
    by_module = {}
    for call in calls:
        header, *offsets = [line for line in call.splitlines() if line]
        assert "--demangle" in header and "--output-style=JSON" in header
        by_module[header.split("--obj=")[1].split()[0]] = set(offsets)
    assert by_module == {"/lib/a.so": {"0x10", "0x30"}, "/lib/b.so": {"0x20"}}


def test_an_attributed_frame_is_not_symbolised(tmp_path):
    """Both frame shapes, and in_circt_object."""
    tool = "/workspace/circt/build/bin/circt-opt"
    frames = [
        {"index": 0, "address": "0x1", "shape": "attributed", "module": "",
         "offset": "", "function": "parseHWArray(mlir::AsmParser&)",
         "file": "/workspace/circt/lib/Dialect/HW/HWTypes.cpp", "line": 723},
        {"index": 1, "address": "0x2", "shape": "attributed", "module": "",
         "offset": "", "function": "generatedTypeParser",
         "file": "/workspace/circt/build/include/circt/HWTypes.cpp.inc", "line": 34},
        {"index": 2, "address": "0x3", "shape": "attributed", "module": "",
         "offset": "", "function": "SignalHandler(int)",
         "file": "Signals.cpp", "line": 0},
        {"index": 3, "address": "0x4", "shape": "module_offset", "module": tool,
         "offset": "0x10", "function": "", "file": "", "line": 0},
        {"index": 4, "address": "0x5", "shape": "module_offset",
         "module": "/workspace/circt/build/lib/libCIRCTHW.so.20",
         "offset": "0x20", "function": "", "file": "", "line": 0},
        {"index": 5, "address": "0x6", "shape": "module_offset",
         "module": "/opt/circt-sdk/lib/libMLIRAsmParser.so",
         "offset": "0x30", "function": "", "file": "", "line": 0},
    ]
    out = call_node(circt_symbolize, frames, tool_path=tool,
                    symbolizer=str(tmp_path / "absent"))
    for before, after in zip(frames[:3], out[:3]):
        assert (after["function"], after["file"], after["line"]) == \
            (before["function"], before["file"], before["line"])
    assert [f["in_circt_object"] for f in out] == \
        [True, True, False, True, True, False]
