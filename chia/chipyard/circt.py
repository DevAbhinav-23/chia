"""General-purpose CIRCT chia nodes — firtool lowering, circt-opt invocation, custom-pass rebuild, ninja builds, lit runs, plus async MCP tool wrappers — usable from any chia loop.

These primitives wrap the CIRCT toolchain (firtool, circt-opt) plus chipyard's
Chisel elaboration target so that any CHIA user can compose a CIRCT-driven flow
without re-implementing subprocess glue. They run as ``@ChiaFunction`` tasks on
a worker with the ``circt`` resource (advertised by the chia-circt image, which
ships /opt/circt-sdk and a source tree at /workspace/circt).
``chisel_elaborate_to_chirrtl`` additionally needs Java + sbt; pair it with a chia-chipyard worker
or layer chipyard's build deps into chia-circt if you need it co-located.

Usage example::

    fir = chisel_elaborate_to_chirrtl(chipyard_path, "RocketConfig")["fir_text"]
    hw  = firtool_lower_chirrtl_to_hw(fir, repl_seq_mem=True)["hw_mlir"]
    sv  = circt_opt_lower_hw_to_verilog(hw)["verilog"]
"""

import fnmatch
import json
import logging
import os
import select
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from chia.base.ChiaFunction import ChiaFunction
from chia.base.tools.AsyncJobTool import AsyncJobTool

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tool paths
# --------------------------------------------------------------------------- #
# Prefer the source-tree build of circt-opt (under /workspace/circt/build/bin)
# so that any custom pass linked in via rebuild_circt_opt_with_custom_pass()
# takes effect. Fall back to the prebuilt SDK binary otherwise. firtool lives
# in the SDK only — there is no source-tree alternative.
_CIRCT_SDK_DIR = "/opt/circt-sdk"
_CIRCT_SOURCE_BUILD_BIN = "/workspace/circt/build/bin/circt-opt"
_CIRCT_SOURCE_TREE = "/workspace/circt"
_FIRTOOL_BIN = os.path.join(_CIRCT_SDK_DIR, "bin", "firtool")


def _resolve_circt_opt() -> str:
    """Return the best available circt-opt path.

    Source-tree build wins so that custom passes linked via
    :func:`rebuild_circt_opt_with_custom_pass` are picked up automatically.
    """
    if os.path.isfile(_CIRCT_SOURCE_BUILD_BIN) and os.access(_CIRCT_SOURCE_BUILD_BIN, os.X_OK):
        return _CIRCT_SOURCE_BUILD_BIN
    sdk = os.path.join(_CIRCT_SDK_DIR, "bin", "circt-opt")
    if os.path.isfile(sdk) and os.access(sdk, os.X_OK):
        return sdk
    # Last resort — let subprocess find it on PATH (and surface a clean error).
    return "circt-opt"


# --------------------------------------------------------------------------- #
# firtool: CHIRRTL .fir -> HW MLIR (+ optional mems.conf)
# --------------------------------------------------------------------------- #
@ChiaFunction(resources={"circt": 1})
def firtool_lower_chirrtl_to_hw(
    chirrtl_text: str,
    repl_seq_mem: bool = True,
    extra_args: tuple[str, ...] = (),
    timeout_seconds: int = 600,
) -> dict[str, str]:
    """Lower CHIRRTL FIRRTL text to HW-dialect MLIR via firtool.

    Wraps ``firtool --ir-hw [--repl-seq-mem --repl-seq-mem-file=<path>]``
    against ``/opt/circt-sdk/bin/firtool``. With ``repl_seq_mem=True`` the
    sequential memories are split off into a ``.mems.conf`` file (the same
    file Chipyard's MacroCompiler consumes) and that text is returned in the
    ``mems_conf`` key; otherwise it is the empty string.

    Args:
        chirrtl_text: Contents of a ``.fir`` (CHIRRTL) file.
        repl_seq_mem: Pass ``--repl-seq-mem --repl-seq-mem-file=...`` so
            seq-mems become external instances + a mems.conf sidecar.
        extra_args: Additional CLI flags appended verbatim to the firtool
            command line (e.g. ``("--disable-all-randomization",)``).
        timeout_seconds: subprocess timeout.

    Returns:
        Dict with keys ``hw_mlir`` (transformed MLIR text), ``mems_conf``
        (empty if ``repl_seq_mem=False``), ``stdout``, ``stderr``,
        ``returncode``, and ``success``.

    Example::

        out = firtool_lower_chirrtl_to_hw(open("Top.fir").read())
        assert out["success"]; print(out["hw_mlir"][:200])
    """
    work_dir = tempfile.mkdtemp(prefix="firtool_lower_")
    try:
        fir_path = os.path.join(work_dir, "input.fir")
        mems_conf_path = os.path.join(work_dir, "input.mems.conf")
        with open(fir_path, "w") as f:
            f.write(chirrtl_text)

        cmd: list[str] = [_FIRTOOL_BIN, fir_path, "--ir-hw"]
        if repl_seq_mem:
            cmd += ["--repl-seq-mem", f"--repl-seq-mem-file={mems_conf_path}"]
        cmd += list(extra_args)

        logger.info(f"[firtool] {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            return {
                "hw_mlir": "",
                "mems_conf": "",
                "stdout": (e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")),
                "stderr": (e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")) + f"\nfirtool timed out after {timeout_seconds}s",
                "returncode": "-1",
                "success": "False",
            }

        mems_conf = ""
        if repl_seq_mem and os.path.isfile(mems_conf_path):
            with open(mems_conf_path) as f:
                mems_conf = f.read()

        return {
            "hw_mlir": result.stdout,
            "mems_conf": mems_conf,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": str(result.returncode),
            "success": str(result.returncode == 0),
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# circt-opt: run an arbitrary pass pipeline
# --------------------------------------------------------------------------- #
@ChiaFunction(resources={"circt": 1})
def circt_opt_run(
    mlir_text: str,
    pass_pipeline: str,
    extra_args: tuple[str, ...] = (),
    timeout_seconds: int = 600,
) -> dict[str, str]:
    """Run ``circt-opt --pass-pipeline='<pipeline>'`` on MLIR text.

    Thin wrapper that prefers ``/workspace/circt/build/bin/circt-opt`` (so
    custom passes linked via :func:`rebuild_circt_opt_with_custom_pass` are
    picked up) and falls back to ``/opt/circt-sdk/bin/circt-opt``.

    Args:
        mlir_text: Input MLIR (any dialect circt-opt understands).
        pass_pipeline: Pipeline string passed verbatim to ``--pass-pipeline``
            (e.g. ``"builtin.module(hw.module(canonicalize))"``).
        extra_args: Additional CLI flags appended after the pipeline.
        timeout_seconds: subprocess timeout.

    Returns:
        Dict with keys ``stdout`` (transformed MLIR), ``stderr``,
        ``returncode``, and ``success``.

    Example::

        out = circt_opt_run(hw_mlir, "builtin.module(canonicalize)")
        print(out["stdout"])
    """
    circt_opt = _resolve_circt_opt()
    cmd: list[str] = [circt_opt, f"--pass-pipeline={pass_pipeline}"]
    cmd += list(extra_args)
    logger.info(f"[circt-opt] {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, input=mlir_text, capture_output=True, text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        return {
            "stdout": (e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")),
            "stderr": (e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")) + f"\ncirct-opt timed out after {timeout_seconds}s",
            "returncode": "-1",
            "success": "False",
        }
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": str(result.returncode),
        "success": str(result.returncode == 0),
    }


# --------------------------------------------------------------------------- #
# circt-opt: HW -> SystemVerilog
# --------------------------------------------------------------------------- #
@ChiaFunction(resources={"circt": 1})
def circt_opt_lower_hw_to_verilog(
    hw_mlir_text: str,
    extra_args: tuple[str, ...] = (),
    timeout_seconds: int = 600,
) -> dict[str, str]:
    """Lower HW-dialect MLIR to SystemVerilog via firtool's ``--ir-hw -> sv`` path.

    Convenience wrapper that runs firtool with ``--format=mlir`` to consume
    HW MLIR and emit Verilog with the standard lowering options applied
    (mirrors what chipyard's flow uses by default).

    Args:
        hw_mlir_text: HW-dialect MLIR (as produced by
            :func:`firtool_lower_chirrtl_to_hw`).
        extra_args: Additional CLI flags appended verbatim.
        timeout_seconds: subprocess timeout.

    Returns:
        Dict with keys ``verilog`` (emitted SV text on success, empty
        otherwise), ``stdout``, ``stderr``, ``returncode``, ``success``.

    Example::

        out = circt_opt_lower_hw_to_verilog(hw_mlir)
        print(out["verilog"][:200])
    """
    work_dir = tempfile.mkdtemp(prefix="hw_to_sv_")
    try:
        in_path = os.path.join(work_dir, "input.hw.mlir")
        out_path = os.path.join(work_dir, "out.sv")
        with open(in_path, "w") as f:
            f.write(hw_mlir_text)

        lowering = (
            "disallowLocalVariables,disallowPackedArrays,"
            "locationInfoStyle=wrapInAtSquareBracket"
        )
        cmd: list[str] = [
            _FIRTOOL_BIN, in_path, "--format=mlir",
            f"--lowering-options={lowering}",
            "-o", out_path,
        ]
        cmd += list(extra_args)
        logger.info(f"[firtool hw->sv] {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            return {
                "verilog": "",
                "stdout": (e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")),
                "stderr": (e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")) + f"\nfirtool hw->sv timed out after {timeout_seconds}s",
                "returncode": "-1",
                "success": "False",
            }

        verilog = ""
        if result.returncode == 0 and os.path.isfile(out_path):
            with open(out_path) as f:
                verilog = f.read()

        return {
            "verilog": verilog,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": str(result.returncode),
            "success": str(result.returncode == 0),
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Rebuild circt-opt with a user-provided custom pass
# --------------------------------------------------------------------------- #
@ChiaFunction(resources={"circt": 1})
def rebuild_circt_opt_with_custom_pass(
    pass_cpp_path: str,
    td_snippet_path: str | None = None,
    transforms_subdir: str = "lib/Dialect/HW/Transforms",
    passes_td_path: str = "include/circt/Dialect/HW/Passes.td",
    num_cpus: int = 16,
    timeout_seconds: int = 3600,
) -> dict[str, str]:
    """Drop a user ``.cpp`` into the CIRCT source tree, patch ``Passes.td``, rebuild ``circt-opt``.

    1. Copy ``pass_cpp_path`` into ``/workspace/circt/<transforms_subdir>/``.
    2. Append the filename to the matching ``CMakeLists.txt`` if not present.
    3. If ``td_snippet_path`` is given, splice it into ``Passes.td`` right
       before the closing ``#endif`` guard (idempotent).
    4. ``ninja -C /workspace/circt/build circt-opt`` (with ``-j ninja_jobs``
       when nonzero, else ninja's default).

    Args:
        pass_cpp_path: Absolute host path to the ``.cpp`` to inject.
        td_snippet_path: Optional ``.td`` snippet to splice into ``Passes.td``.
            If ``None``, ``Passes.td`` is left untouched (useful when the pass
            already has its own .td definition).
        transforms_subdir: Subdirectory under ``/workspace/circt/`` where the
            ``.cpp`` should land (and its ``CMakeLists.txt`` lives). Defaults
            to the HW Transforms dir.
        passes_td_path: Path under ``/workspace/circt/`` to the ``Passes.td``
            file to patch.
        num_cpus: ``-j`` value for ninja. Default 16 keeps headroom on a
            32-core host. Pair with ``.options(num_cpus=N)`` at the call
            site if you also want Ray to reserve N CPUs per dispatch.
        timeout_seconds: subprocess timeout for the ninja build.

    Returns:
        Dict with keys ``success`` (``"True"``/``"False"``), ``log``
        (combined stdout/stderr from ninja), and ``binary_path`` (the
        rebuilt ``circt-opt`` location on success, empty otherwise).

    Example::

        r = rebuild_circt_opt_with_custom_pass("/work/passes/MyPass.cpp",
                                               "/work/my_pass_def.td")
        print(r["success"], r["binary_path"])
    """
    if not os.path.isfile(pass_cpp_path):
        return {"success": "False",
                "log": f"pass_cpp_path does not exist: {pass_cpp_path}",
                "binary_path": ""}
    if not os.path.isdir(_CIRCT_SOURCE_TREE):
        return {"success": "False",
                "log": f"CIRCT source tree not found at {_CIRCT_SOURCE_TREE}; "
                       "this primitive requires the chia-circt image with a "
                       "source checkout under /workspace/circt.",
                "binary_path": ""}

    transforms_dir = os.path.join(_CIRCT_SOURCE_TREE, transforms_subdir)
    cmake_lists = os.path.join(transforms_dir, "CMakeLists.txt")
    if not os.path.isdir(transforms_dir):
        return {"success": "False",
                "log": f"transforms_subdir does not exist: {transforms_dir}",
                "binary_path": ""}

    # Step 1: copy the .cpp into the source tree.
    cpp_basename = os.path.basename(pass_cpp_path)
    dest_cpp = os.path.join(transforms_dir, cpp_basename)
    shutil.copyfile(pass_cpp_path, dest_cpp)

    # Step 2: append to CMakeLists.txt if missing.
    cmake_log = ""
    if os.path.isfile(cmake_lists):
        with open(cmake_lists) as f:
            cmake_src = f.read()
        if cpp_basename not in cmake_src:
            # Insert right after the add_circt_dialect_library(... line.
            # Mirrors `sed -i '/^add_circt_dialect_library(...$/a\  Foo.cpp'`
            # in 1_circtpasses/build_pass.sh.
            new_lines: list[str] = []
            inserted = False
            for line in cmake_src.splitlines():
                new_lines.append(line)
                if (not inserted) and line.strip().startswith("add_circt_dialect_library("):
                    new_lines.append(f"  {cpp_basename}")
                    inserted = True
            if inserted:
                with open(cmake_lists, "w") as f:
                    f.write("\n".join(new_lines) + "\n")
                cmake_log = f"appended {cpp_basename} to {cmake_lists}\n"
            else:
                cmake_log = (f"WARNING: could not find add_circt_dialect_library() "
                             f"in {cmake_lists}; {cpp_basename} not registered\n")
        else:
            cmake_log = f"{cpp_basename} already in {cmake_lists}\n"

    # Step 3: optional Passes.td patch (idempotent).
    td_log = ""
    if td_snippet_path is not None:
        if not os.path.isfile(td_snippet_path):
            return {"success": "False",
                    "log": cmake_log + f"td_snippet_path not found: {td_snippet_path}",
                    "binary_path": ""}
        td_full_path = os.path.join(_CIRCT_SOURCE_TREE, passes_td_path)
        if not os.path.isfile(td_full_path):
            return {"success": "False",
                    "log": cmake_log + f"Passes.td not found: {td_full_path}",
                    "binary_path": ""}
        with open(td_snippet_path) as f:
            snippet = f.read()
        with open(td_full_path) as f:
            td_src = f.read()
        # Idempotency: skip if any "def " line from the snippet is already
        # present. Cheap heuristic mirroring patch_passes_td.py.
        snippet_defs = [ln for ln in snippet.splitlines() if ln.strip().startswith("def ")]
        already = any(d.strip() in td_src for d in snippet_defs)
        if already:
            td_log = f"Passes.td already contains snippet defs, skipping inject\n"
        else:
            # Find the closing #endif guard and inject before it.
            lines = td_src.splitlines()
            endif_idx = None
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().startswith("#endif"):
                    endif_idx = i
                    break
            if endif_idx is None:
                return {"success": "False",
                        "log": cmake_log + f"no #endif guard found in {td_full_path}",
                        "binary_path": ""}
            new_td = "\n".join(lines[:endif_idx]) + "\n" + snippet + "\n" + "\n".join(lines[endif_idx:])
            with open(td_full_path, "w") as f:
                f.write(new_td)
            td_log = f"injected {td_snippet_path} into {td_full_path}\n"
            # Touch the file so tablegen reruns.
            os.utime(td_full_path, None)

    # Step 4: ninja rebuild.
    build_dir = os.path.join(_CIRCT_SOURCE_TREE, "build")
    cmd: list[str] = ["ninja", "-C", build_dir]
    if num_cpus > 0:
        cmd += ["-j", str(num_cpus)]
    cmd += ["circt-opt"]
    logger.info(f"[rebuild circt-opt] {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        return {"success": "False",
                "log": cmake_log + td_log + out + err + f"\nninja timed out after {timeout_seconds}s",
                "binary_path": ""}

    combined_log = cmake_log + td_log + result.stdout + result.stderr
    if result.returncode != 0:
        return {"success": "False", "log": combined_log, "binary_path": ""}

    binary_path = os.path.join(build_dir, "bin", "circt-opt")
    return {
        "success": "True",
        "log": combined_log,
        "binary_path": binary_path if os.path.isfile(binary_path) else "",
    }


# --------------------------------------------------------------------------- #
# Discoverability: list available circt-opt passes
# --------------------------------------------------------------------------- #
@ChiaFunction(resources={"circt": 1})
def list_circt_passes(
    category: str | None = None,
    timeout_seconds: int = 60,
) -> list[str]:
    """List available ``circt-opt`` passes, optionally filtered by dialect prefix.

    Runs ``circt-opt --help`` and parses out lines beginning with ``--`` that
    look like pass flags (single-token, no spaces, not a built-in option).

    Args:
        category: Optional pass-name prefix to filter by (e.g. ``"hw-"``,
            ``"firrtl-"``). ``None`` returns every parsed pass.
        timeout_seconds: subprocess timeout.

    Returns:
        Sorted list of pass names (without the leading ``--``).

    Example::

        hw_passes = list_circt_passes(category="hw-")
        print(hw_passes[:5])
    """
    circt_opt = _resolve_circt_opt()
    try:
        result = subprocess.run(
            [circt_opt, "--help"], capture_output=True, text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return []

    # circt-opt --help prints lines like "  --pass-name   - Description".
    # Some lines have "=<arg>" suffixes; strip those.
    passes: set[str] = set()
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith("--"):
            continue
        token = line.split()[0]
        name = token[2:]  # drop leading "--"
        # Strip "=<arg>" suffix on options that take an inline value.
        if "=" in name:
            name = name.split("=", 1)[0]
        if not name or " " in name:
            continue
        passes.add(name)

    out = sorted(passes)
    if category is not None:
        out = [p for p in out if p.startswith(category)]
    return out


# --------------------------------------------------------------------------- #
# Chipyard: elaborate a Chisel config -> CHIRRTL .fir + anno.json
# --------------------------------------------------------------------------- #
@ChiaFunction(resources={"circt": 1})
def chisel_elaborate_to_chirrtl(
    chipyard_path: str,
    config: str,
    config_package: str = "chipyard",
    num_cpus: int = 16,
    timeout_seconds: int = 1800,
) -> dict[str, str]:
    """Elaborate a chipyard Chisel config and return its CHIRRTL ``.fir`` text.

    Drives the chipyard make target that produces the ``.fir`` and
    ``.anno.json`` for a given ``CONFIG`` / ``CONFIG_PACKAGE`` pair, so the
    output is ready to pipe straight into
    :func:`firtool_lower_chirrtl_to_hw`. Runs in ``<chipyard>/sims/verilator``
    (the canonical entry point that re-exports ``firrtl``).

    Args:
        chipyard_path: Path to the chipyard installation.
        config: Chisel config name (e.g. ``"RocketConfig"``).
        config_package: Scala package holding the config (default
            ``"chipyard"``).
        num_cpus: ``-j`` for the make invocation. Default 16 keeps
            headroom on a 32-core host. Pair with ``.options(num_cpus=N)``
            at the call site if you also want Ray to reserve N CPUs per
            dispatch.
        timeout_seconds: subprocess timeout.

    Returns:
        Dict with keys ``fir_text`` (CHIRRTL contents on success, empty
        otherwise), ``anno_json`` (annotations JSON, empty otherwise),
        ``stdout``, ``stderr``, ``returncode``, ``success``.

    Example::

        r = chisel_elaborate_to_chirrtl("/scratch/chipyard", "RocketConfig")
        print(r["fir_text"][:200])
    """
    sims_dir = os.path.join(chipyard_path, "sims/verilator")
    if not os.path.isdir(sims_dir):
        return {
            "fir_text": "", "anno_json": "",
            "stdout": "", "stderr": f"sims/verilator not found at {sims_dir}",
            "returncode": "-1", "success": "False",
        }

    # The `firrtl` make target produces the .fir + .anno.json under
    # generated-src/<package>.harness.TestHarness.<config>/.
    cmd = (
        f"make -j {num_cpus} firrtl "
        f"CONFIG={config} CONFIG_PACKAGE={config_package}"
    )
    logger.info(f"[chisel elaborate] {cmd} (cwd={sims_dir})")
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=sims_dir, capture_output=True, text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        return {
            "fir_text": "", "anno_json": "",
            "stdout": (e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")),
            "stderr": (e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")) + f"\nelaborate timed out after {timeout_seconds}s",
            "returncode": "-1", "success": "False",
        }

    gen_src_dir = os.path.join(
        sims_dir, "generated-src",
        f"{config_package}.harness.TestHarness.{config}",
    )

    # Find the .fir + .anno.json (firtool names them after the top module,
    # not always after the config — scan the dir instead of guessing).
    fir_text = ""
    anno_json = ""
    if os.path.isdir(gen_src_dir):
        for fname in os.listdir(gen_src_dir):
            full = os.path.join(gen_src_dir, fname)
            if fname.endswith(".fir") and not fir_text:
                with open(full) as f:
                    fir_text = f.read()
            elif fname.endswith(".anno.json") and not anno_json:
                with open(full) as f:
                    anno_json = f.read()

    success = result.returncode == 0 and bool(fir_text)
    return {
        "fir_text": fir_text,
        "anno_json": anno_json,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": str(result.returncode),
        "success": str(success),
    }


# --------------------------------------------------------------------------- #
# Source-tree build/test primitives.
#
# Build arbitrary CIRCT targets and run the lit test suite against the
# /workspace/circt checkout. Unlike the string-typed helpers above, these
# return real bool/int/list so callers can branch without parsing. They run
# subprocesses in their own process group and SIGKILL the group on timeout, so
# a lingering compiler/linker/test child that keeps the stdout pipe open can't
# wedge communicate() forever.
# --------------------------------------------------------------------------- #
_CIRCT_BUILD_DIR = os.path.join(_CIRCT_SOURCE_TREE, "build")
_WARM_SENTINEL = os.path.join(_CIRCT_BUILD_DIR, ".chia_circt_warm")
_PY_WORKER_BIN = "/home/ray/anaconda3/envs/py_worker/bin"
_PIP_BIN = os.path.join(_PY_WORKER_BIN, "pip")
_LIT_BIN = os.path.join(_PY_WORKER_BIN, "lit")


def _tail(text: str, n: int = 120) -> str:
    """Last *n* lines of *text* (keeps task return payloads bounded)."""
    return "\n".join(text.splitlines()[-n:])


@ChiaFunction(resources={"circt": 1})
def circt_ninja_build(
    targets: tuple[str, ...] = ("circt-opt",),
    num_cpus: int = 16,
    timeout_seconds: int = 1800,
) -> dict:
    """Build CIRCT *targets* with ``ninja -C /workspace/circt/build [-j N]``.

    Generalises the build step of :func:`rebuild_circt_opt_with_custom_pass` to
    any target set (``circt-opt``, ``firtool``, ``arcilator``, ...). Incremental:
    only the touched objects + the affected tools relink.

    Runs ninja in its OWN process group (``start_new_session``) and SIGKILLs the
    whole group on timeout. Without this, a lingering child (clang/lld) that keeps
    the stdout pipe open wedges ``communicate()`` forever — even after ninja
    exits. Returns ``{success: bool, returncode: int, log_tail: str}``.
    """
    cmd = ["ninja", "-C", _CIRCT_BUILD_DIR]
    if num_cpus > 0:
        cmd += ["-j", str(num_cpus)]
    cmd += list(targets)
    logger.info(f"[ninja] {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True)
    except Exception as e:  # noqa: BLE001
        return {"success": False, "returncode": -1, "log_tail": f"failed to spawn ninja: {e}"}
    try:
        out, _ = proc.communicate(timeout=timeout_seconds)
        return {"success": proc.returncode == 0, "returncode": proc.returncode,
                "log_tail": _tail(out or "")}
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # kill ninja + all compiler/linker children
        except ProcessLookupError:
            pass
        try:
            out, _ = proc.communicate(timeout=30)
        except Exception:  # noqa: BLE001
            out = ""
        return {"success": False, "returncode": -1,
                "log_tail": _tail((out or "") + f"\nninja killed after {timeout_seconds}s timeout")}


@ChiaFunction(resources={"circt": 1})
def circt_warm_build(
    targets: tuple[str, ...] = ("circt-opt", "firtool"),
    num_cpus: int = 16,
    timeout_seconds: int = 5400,
) -> dict:
    """Idempotent per-container warm-up: ensure ``lit`` is installed and *targets*
    are built, then drop a sentinel so repeat calls are no-ops.

    The chia-circt image bakes only ``circt-opt``; warming the other tool targets
    here is cheap (shared dialect libs are already built). Safe to call at the
    start of every task — only the first call on a given container does work.

    Returns ``{success: bool, warmed: bool, log_tail: str}`` (``warmed=False``
    means the sentinel already existed).
    """
    if os.path.isfile(_WARM_SENTINEL):
        return {"success": True, "warmed": False, "log_tail": "already warm"}
    log: list[str] = []
    if not os.path.isfile(_LIT_BIN):
        try:
            r = subprocess.run([_PIP_BIN, "install", "--quiet", "lit"],
                               capture_output=True, text=True, timeout=600)
            log.append(f"pip install lit: rc={r.returncode}")
        except subprocess.TimeoutExpired:
            log.append("pip install lit timed out")
    build = circt_ninja_build(tuple(targets), num_cpus=num_cpus, timeout_seconds=timeout_seconds)
    log.append(build["log_tail"])
    if build["success"]:
        try:
            with open(_WARM_SENTINEL, "w") as f:
                f.write("warm\n")
        except OSError as e:
            log.append(f"could not write sentinel: {e}")
    return {"success": build["success"], "warmed": True, "log_tail": _tail("\n".join(log))}


@ChiaFunction(resources={"circt": 1})
def circt_run_lit(
    test_paths: tuple[str, ...],
    extra_args: tuple[str, ...] = (),
    timeout_seconds: int = 1800,
    filter_out: str = "",
) -> dict:
    """Run ``lit`` on build/test path(s) (relative to /workspace/circt/build).

    Resolves the py_worker ``lit`` (installed by :func:`circt_warm_build`), falls
    back to PATH. FileCheck/not/count come from the SDK on PATH. Empty
    ``test_paths`` → success with zero tests (nothing to regress). ``filter_out``
    is a regex passed to lit's ``--filter-out`` to skip matching test names.

    Returns ``{success: bool, passed: int, failed: int, failures: list[str], log_tail: str}``.
    """
    import re
    paths = [p for p in test_paths if p]
    if not paths:
        return {"success": True, "passed": 0, "failed": 0, "failures": [], "log_tail": "no test paths"}
    lit = _LIT_BIN if os.path.isfile(_LIT_BIN) else (shutil.which("lit") or "lit")
    filt = [f"--filter-out={filter_out}"] if filter_out else []
    cmd = [lit, "--no-progress-bar", *filt, *extra_args, *paths]
    logger.info(f"[lit] {' '.join(cmd)}")
    # Own process group + SIGKILL the group on timeout, so a hung test can't wedge
    # communicate() forever (same hazard, and fix, as circt_ninja_build).
    try:
        proc = subprocess.Popen(cmd, cwd=_CIRCT_BUILD_DIR, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, start_new_session=True)
    except Exception as e:  # noqa: BLE001
        return {"success": False, "passed": 0, "failed": -1, "failures": ["<spawn-failed>"],
                "log_tail": f"failed to spawn lit: {e}"}
    try:
        out, _ = proc.communicate(timeout=timeout_seconds)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            out, _ = proc.communicate(timeout=30)
        except Exception:  # noqa: BLE001
            out = ""
        return {"success": False, "passed": 0, "failed": -1, "failures": ["<timeout>"],
                "log_tail": _tail((out or "") + f"\nlit killed after {timeout_seconds}s timeout")}
    out = out or ""
    failures = [ln.split(":", 1)[1].strip() for ln in out.splitlines()
                if ln.startswith(("FAIL:", "UNRESOLVED:", "TIMEOUT:", "XPASS:"))]

    def _count(label: str) -> int:
        # lit pads its summary labels to align colons when several result
        # categories are present (e.g. "Passed           : 197" alongside
        # "Expectedly Failed:   3"), so allow whitespace around the colon. Match
        # at the start of the stripped line so "Failed" does NOT also catch the
        # "Expectedly Failed" / "Unexpectedly Failed" (XFAIL/XPASS) lines.
        for ln in out.splitlines():
            m = re.match(rf"{label}\s*:\s*(\d+)", ln.strip())
            if m:
                return int(m.group(1))
        return 0

    return {"success": rc == 0, "passed": _count("Passed"),
            "failed": _count("Failed"), "failures": failures,
            "log_tail": _tail(out)}


# --------------------------------------------------------------------------- #
# MCP ChiaTool wrappers
# --------------------------------------------------------------------------- #
# Thin async MCP tools over circt_ninja_build / circt_run_lit, for handing to an
# LLM agent so it gets crisp, parsed build/test signals instead of scraping raw
# bash output. Pin each to the container that hosts the CIRCT checkout via
# task_options (NodeAffinity). Build and lit are ASYNC (start + poll) on the
# canonical AsyncJobTool base: a multi-minute *synchronous* MCP call can lose its
# response on the streamable-HTTP transport and hang the agent, so each call
# returns quickly — `*_<verb>` starts the job and returns immediately, and
# `*_status` long-polls in short, bounded chunks until done=true.


class BuildTool(AsyncJobTool):
    """MCP tool: rebuild CIRCT tool target(s) with ninja, ASYNCHRONOUSLY."""

    def __init__(self, name: str, num_cpus: int = 16, task_options=None):
        super().__init__(name, task_options=task_options)
        self._jobs = num_cpus
        self.mcp.add_tool(self.build, name=f"{name}_build")
        self.mcp.add_tool(self.build_status, name=f"{name}_build_status")
        super().__post_init__()

    def build(self, targets: list[str] | None = None) -> dict:
        """Start a ninja build of CIRCT target(s) in the BACKGROUND and return
        IMMEDIATELY (does NOT wait). Then poll `<name>_build_status` until it
        returns done=true to get {success, returncode, log_tail}.

        Args:
            targets: e.g. ["circt-opt"] (default) or ["firtool"].
        """
        targets = list(targets) if targets else ["circt-opt"]
        r = self._job_start(lambda: circt_ninja_build(tuple(targets), num_cpus=self._jobs))
        r["targets"] = targets
        if r.get("started"):
            r["note"] = "build started in background; poll build_status until done=true"
        return r

    def build_status(self, wait_seconds: int = 60) -> dict:
        """Await the in-flight build (blocks up to wait_seconds, capped at 120).
        Returns done=true with {success, returncode, log_tail} once finished."""
        return self._job_status(wait_seconds)


class LitTool(AsyncJobTool):
    """MCP tool: run lit regression tests, ASYNCHRONOUSLY (same rationale as build —
    a whole-dialect lit run can take minutes)."""

    def __init__(self, name: str, task_options=None):
        super().__init__(name, task_options=task_options)
        self.mcp.add_tool(self.run_lit, name=f"{name}_run_lit")
        self.mcp.add_tool(self.lit_status, name=f"{name}_lit_status")
        super().__post_init__()

    def run_lit(self, test_paths: list[str]) -> dict:
        """Start a lit run on build/test path(s) (e.g. ["test/Dialect/Comb"]) in the
        BACKGROUND and return IMMEDIATELY. Then poll `<name>_lit_status` until it
        returns done=true to get {success, passed, failed, failures, log_tail}.
        """
        paths = list(test_paths)
        r = self._job_start(lambda: circt_run_lit(tuple(paths)))
        r["test_paths"] = paths
        if r.get("started"):
            r["note"] = "lit started in background; poll lit_status until done=true"
        return r

    def lit_status(self, wait_seconds: int = 60) -> dict:
        """Await the in-flight lit run (blocks up to wait_seconds, capped at 120).
        Returns done=true with the parsed lit result once finished."""
        return self._job_status(wait_seconds)

# --------------------------------------------------------------------------- #
# Probe, reduce and symbolise: run one CIRCT tool on one input under hard
# limits, run circt-reduce with an interestingness script, and resolve an LLVM
# crash trace against the objects its frames lie in. Generic: they know nothing
# about the flow that calls them.
# --------------------------------------------------------------------------- #

#: The gap between `prlimit --cpu`'s soft and hard values (a SIGXCPU at the soft value, SIGKILL at soft + this).
CPU_HARD_MARGIN_SECONDS = 5

#: The allocation-failure texts that mark a probe as out of memory.
ALLOCATION_FAILURE_LITERALS = ("std::bad_alloc", "out of memory",
                               "LLVM ERROR: out of memory")

#: `RLIMIT_AS` does not kill.
_MEMORY_ERROR_LINE = "MemoryError"

#: The same failures as the LINE OPENINGS a failing runtime actually prints.
ALLOCATION_FAILURE_LINES = (
    "terminate called after throwing an instance of 'std::bad_alloc'",
    "what():  std::bad_alloc",
    "LLVM ERROR: out of memory",
    "out of memory",
    _MEMORY_ERROR_LINE,
)

#: Where the chia-circt images put the source-built binaries.
CIRCT_BIN_DIR = "/workspace/circt/build/bin"

#: The `in_circt_object` default, which is the image's one CIRCT root.
CIRCT_ROOTS = ("/workspace/circt/",)

#: How long the drain is allowed to continue after the wall-clock killer fires.
_POST_KILL_DRAIN_SECONDS = 2.0

#: The select() slice, which bounds how late the wall-clock killer can fire.
_POLL_SECONDS = 0.2


@ChiaFunction(resources={"circt": 1})
def circt_exec_probe(tool: str, argv: list, *, cwd: str,
                     wall_seconds: int, address_space_bytes: int,
                     cpu_seconds: int, nofile: int = 1024,
                     output_byte_cap: int = 1_000_000) -> dict:
    """Run one CIRCT binary on one input, bounded, with no shell anywhere.

    Returns:
        {"exit_status": int | None, "signal": str | None, "limit_hit": str | None, "cpu_seconds": float, "peak_rss_bytes": int, "wall_seconds": float, "stdout": str, "stderr": str, "truncated": bool, "argv": list[str], "worker_hostname": str, "worker_node_id": str, "child_pid": int}.
    Worker:
        {"circt": 1}.
    Raises:
        FileNotFoundError if tool does not exist; every other failure is a field.
    """
    if not os.path.isfile(tool):
        raise FileNotFoundError(tool)
    if not os.path.isdir(cwd):
        raise FileNotFoundError(cwd)
    full = ["prlimit",
            f"--as={address_space_bytes}",
            f"--cpu={cpu_seconds}:{cpu_seconds + CPU_HARD_MARGIN_SECONDS}",
            f"--nofile={nofile}",
            "--",
            tool, *argv]

    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()
    started = time.monotonic()
    pid = os.fork()
    if pid == 0:                                    # the child; never returns
        try:
            os.setsid()
            os.dup2(out_w, 1)
            os.dup2(err_w, 2)
            for fd in (out_r, out_w, err_r, err_w):
                os.close(fd)
            os.chdir(cwd)
            os.execvp(full[0], full)
        except BaseException:                       # pragma: no cover - exec failed
            os._exit(127)
    os.close(out_w)
    os.close(err_w)

    deadline = started + wall_seconds
    try:
        streams, over_cap, killed = _drain({1: out_r, 2: err_r}, deadline,
                                           output_byte_cap, pid)
        status, rusage, killed = _reap(pid, deadline, killed)
    finally:
        _killpg(pid)
    wall = time.monotonic() - started

    returncode = -os.WTERMSIG(status) if os.WIFSIGNALED(status) else os.WEXITSTATUS(status)
    sig = _signal_name(-returncode) if returncode < 0 else None
    exit_status = returncode if returncode >= 0 else None
    stderr_text = streams[2]
    cpu_used = rusage.ru_utime + rusage.ru_stime
    peak_rss = rusage.ru_maxrss * 1024              # ru_maxrss is kB on Linux
    return {"exit_status": exit_status,
            "signal": sig,
            "limit_hit": _limit_hit(killed, sig, cpu_used, cpu_seconds, stderr_text,
                                    peak_rss_bytes=peak_rss,
                                    address_space_bytes=address_space_bytes),
            "cpu_seconds": cpu_used,
            "peak_rss_bytes": peak_rss,
            "wall_seconds": wall,
            "stdout": streams[1],
            "stderr": stderr_text,
            "truncated": over_cap,
            "argv": full,
            "worker_hostname": socket.gethostname(),
            "worker_node_id": _node_id(),
            "child_pid": pid}


@ChiaFunction(resources={"circt": 1})
def circt_reduce_run(input_path: str, test_script: str, output_path: str, *,
                     test_args: tuple = (), wall_seconds: int = 60,
                     sigkill_grace_seconds: int = 10,
                     binary: str = CIRCT_BIN_DIR + "/circt-reduce") -> dict:
    """Run circt-reduce with an interestingness script and an output path.

    --test-must-fail is never passed: the script's polarity is exit 0 for
    interesting. `output_valid` is computed once the process has
    exited, because --keep-best defaults to true and a kill can land mid-write.

    Returns:
        {"success": bool, "returncode": int, "wall_seconds": float,
         "output_valid": bool, "log_tail": str}
    Worker:
        {"circt": 1}.
    Raises:
        nothing.
    """
    argv = [binary, input_path, f"--test={test_script}",
            *(f"--test-arg={a}" for a in test_args),
            "--keep-best", "-o", output_path]
    started = time.monotonic()
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        log = proc.communicate(timeout=wall_seconds)[0]
    except subprocess.TimeoutExpired:
        _killpg(proc.pid, signal.SIGTERM)
        try:
            log = proc.communicate(timeout=sigkill_grace_seconds)[0]
        except subprocess.TimeoutExpired:
            _killpg(proc.pid, signal.SIGKILL)
            log = proc.communicate()[0]
    finally:
        _killpg(proc.pid)
    out = Path(output_path)
    return {"success": proc.returncode == 0,
            "returncode": proc.returncode,
            "wall_seconds": time.monotonic() - started,
            "output_valid": out.is_file() and out.stat().st_size > 0,
            "log_tail": (log or b"").decode("utf-8", "backslashreplace")[-4000:]}


@ChiaFunction(resources={"circt": 1})
def circt_symbolize(frames: list, *, tool_path: str = "",
                    circt_roots: tuple = CIRCT_ROOTS,
                    symbolizer: str = "llvm-symbolizer",
                    timeout_seconds: int = 120) -> list:
    """Resolve LLVM crash-trace frames against the objects they actually lie in.

    Returns:
        one dict per input frame, in input order, with the same keys plus "in_circt_object".
    Worker:
        {"circt": 1} - the objects are the worker's.
    Raises:
        nothing.
    """
    out = [dict(f) for f in frames]
    by_module: dict = {}
    for i, frame in enumerate(out):
        if frame.get("shape") == "module_offset" and frame.get("module"):
            by_module.setdefault(frame["module"], []).append(i)
    for module, indexes in by_module.items():
        records = _symbolize_module(symbolizer, module,
                                    [out[i].get("offset", "") for i in indexes],
                                    timeout_seconds)
        for i, record in zip(indexes, records):
            # MERGE, never replace.
            out[i].update({key: value for key, value in record.items() if value})
    for frame in out:
        frame["in_circt_object"] = _in_circt_object(frame, tool_path, circt_roots)
    return out


# --- the parts, which are not nodes and take no worker resource -------------

def _symbolize_module(symbolizer: str, module: str, offsets: list,
                      timeout_seconds: int) -> list:
    """One llvm-symbolizer call for one module; one record per offset, in order."""
    blank = {"function": "", "file": "", "line": 0}
    if not offsets:
        return []
    try:
        proc = subprocess.run(
            [symbolizer, f"--obj={module}", "--demangle", "--output-style=JSON"],
            input="\n".join(offsets) + "\n", capture_output=True, text=True,
            timeout=timeout_seconds)
    except (OSError, subprocess.SubprocessError):
        return [dict(blank) for _ in offsets]
    records = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            symbol = (json.loads(line).get("Symbol") or [{}])[0]
        except (ValueError, AttributeError, IndexError):
            records.append(dict(blank))
            continue
        records.append({"function": symbol.get("FunctionName", "") or "",
                        "file": symbol.get("FileName", "") or "",
                        "line": int(symbol.get("Line") or 0)})
    records.extend(dict(blank) for _ in range(len(offsets) - len(records)))
    return records[:len(offsets)]


def _in_circt_object(frame: dict, tool_path: str, circt_roots: tuple) -> bool:
    """§3.6.2's predicate, one line per shape."""
    if frame.get("shape") == "module_offset":
        module = frame.get("module", "")
        return bool(module) and (module == tool_path
                                 or fnmatch.fnmatch(os.path.basename(module),
                                                    "libCIRCT*.so*"))
    path = frame.get("file", "")
    return bool(path) and any(path.startswith(root) for root in circt_roots)


def allocation_evidence(stderr: str) -> bool:
    """Whether *stderr* REPORTS an allocation failure, rather than mentioning one."""
    return any(line.strip().startswith(ALLOCATION_FAILURE_LINES)
               for line in stderr.splitlines())


def _limit_hit(killed: bool, sig: Optional[str], cpu_used: float,
               cpu_seconds: int, stderr: str, *, peak_rss_bytes: int = 0,
               address_space_bytes: int = 0) -> Optional[str]:
    """§3.10's three rules, tested in this order, and the only place they live."""
    if killed:
        return "wall"
    if sig == "SIGXCPU" or cpu_used >= cpu_seconds:
        return "cpu"
    if allocation_evidence(stderr):
        return "address_space"
    if address_space_bytes and peak_rss_bytes >= address_space_bytes:
        return "address_space"
    return None


def _drain(fds: dict, deadline: float, cap: int, pgid: int) -> tuple:
    """Read both streams to EOF or to *deadline*, killing the group at expiry."""
    buffers = {key: bytearray() for key in fds}
    seen = {key: 0 for key in fds}
    open_fds = {fd: key for key, fd in fds.items()}
    killed, grace_end = False, 0.0
    try:
        while open_fds:
            now = time.monotonic()
            if not killed and now >= deadline:
                _killpg(pgid)
                killed, grace_end = True, now + _POST_KILL_DRAIN_SECONDS
            if killed and now >= grace_end:
                break
            budget = (grace_end if killed else deadline) - now
            ready = select.select(list(open_fds), [], [],
                                  max(0.0, min(budget, _POLL_SECONDS)))[0]
            for fd in ready:
                chunk = os.read(fd, 65536)
                if not chunk:
                    del open_fds[fd]
                    continue
                key = open_fds[fd]
                seen[key] += len(chunk)
                room = cap - len(buffers[key])
                if room > 0:
                    buffers[key] += chunk[:room]
    finally:
        for fd in fds.values():
            try:
                os.close(fd)
            except OSError:                         # pragma: no cover
                pass
    # backslashreplace, never replace.
    return ({key: bytes(buf).decode("utf-8", "backslashreplace")
             for key, buf in buffers.items()},
            any(seen[key] > cap for key in fds),
            killed)


def _reap(pid: int, deadline: float, killed: bool) -> tuple:
    """os.wait4 the child, killing the group if it outlives *deadline* anyway."""
    while True:
        done, status, rusage = os.wait4(pid, os.WNOHANG)
        if done:
            return status, rusage, killed
        if not killed and time.monotonic() >= deadline:
            _killpg(pid)
            killed = True
        time.sleep(0.01)


def _signal_name(number: int) -> str:
    """"SIGABRT" for 6, and the bare number for a signal outside the enum."""
    try:
        return signal.Signals(number).name
    except ValueError:                              # pragma: no cover
        return f"SIG{number}"


def _killpg(pgid: int, sig: int = signal.SIGKILL) -> None:
    """Signal the whole group, and say nothing when it has already gone."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _node_id() -> str:
    """This Ray node's id, or "" outside a session."""
    ray = sys.modules.get("ray")
    if ray is None or not ray.is_initialized():
        return ""
    try:
        return ray.get_runtime_context().get_node_id()
    except Exception:                               # pragma: no cover
        return ""
