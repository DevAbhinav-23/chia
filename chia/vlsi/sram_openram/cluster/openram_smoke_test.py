"""End-to-end smoke test for the chia-openram worker.

Drives the OpenRAM SRAM pipeline against the REAL compiler + sky130 PDK in
the ``ghcr.io/ucb-bar/chia-openram`` container and validates the collateral
with OpenROAD (``read_liberty`` / ``read_lef`` / ``read_verilog``), headlessly:

  1. ``characterize_top_mems_conf_with_openram`` dispatched from the host
     driver with ``parallel=True``: every distinct macro runs as its own
     ``run_openram`` task on the ``openram`` resource (scheduling path, the
     trampoline, dataclass serialization across the worker boundary).
  2. A second characterization of the same conf must be served entirely from
     the worker-side cache (``from_cache`` on every macro).
  3. On a worker: every corner Liberty and the LEF of every macro, plus the
     macro blackbox stubs, are read by OpenROAD. (The adapters use ``~`` and
     concatenation in port connections, which OpenSTA's structural Verilog
     reader rejects, so they are only checked by Genus in a real synthesis.)
  4. ``prefill_openram_cache`` / ``openram_cache_inventory`` /
     ``export_openram_cache`` / ``import_openram_cache`` round trip.

Small geometries keep the run to a few minutes; see SAMPLE_MEMS_CONF.

Usage (host, chia env active, cluster up via openram_local.yaml):
    cd <repo root>
    python chia/vlsi/sram_openram/cluster/openram_smoke_test.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 4))))

import ray

from chia.base.ChiaFunction import ChiaFunction, get

# 1rw unmasked, 1rw masked, 1r1w masked (the BOOM port kinds), one tiny synflop.
SAMPLE_MEMS_CONF = "\n".join([
    "name small_rw      depth 64  width 16 ports rw",
    "name small_mrw     depth 32  width 16 ports mrw mask_gran 8",
    "name small_1r1w    depth 64  width 8  ports mwrite,read mask_gran 8",
    "name tiny_q        depth 8   width 8  ports rw",   # < THRESHOLD -> stays synflop
])
# The samples are deliberately tiny (a few minutes each); lower the synflop
# threshold so they are characterized at all (the default is 256 B).
THRESHOLD = 16

_TECH_LEF = """\
VERSION 5.7 ;
BUSBITCHARS "[]" ;
DIVIDERCHAR "/" ;
UNITS
  DATABASE MICRONS 2000 ;
END UNITS
LAYER met1
  TYPE ROUTING ;
END met1
LAYER met2
  TYPE ROUTING ;
END met2
LAYER met3
  TYPE ROUTING ;
END met3
LAYER met4
  TYPE ROUTING ;
END met4
END LIBRARY
"""


@ChiaFunction(resources={"openram": 0.01})
def validate_collateral(sram_libs: list[dict], verilog_files: list[tuple[str, str]]) -> dict:
    """On a worker (where OpenROAD lives): read every lib/lef/verilog."""
    import os
    import shutil
    import subprocess
    import tempfile

    tool = shutil.which("openroad")
    report = {"openroad": tool, "libs": {}, "lefs": {}, "verilog": {}}
    if not tool:
        return report

    def _run(script):
        with tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False) as f:
            f.write(script)
            path = f.name
        r = subprocess.run([tool, "-no_init", "-exit", path], capture_output=True, text=True, timeout=300)
        os.unlink(path)
        return r.returncode == 0, (r.stdout + r.stderr)[-500:]

    with tempfile.TemporaryDirectory() as d:
        techp = os.path.join(d, "tech.lef")
        with open(techp, "w") as f:
            f.write(_TECH_LEF)
        for lp in sram_libs:
            for suffix, text in sorted(lp["lib_contents"].items()):
                p = os.path.join(d, f"{lp['name']}_{suffix}.lib")
                with open(p, "w") as f:
                    f.write(text)
                ok, out = _run(f"read_liberty {p}\nexit 0\n")
                report["libs"][f"{lp['name']}:{suffix}"] = (ok, None if ok else out)
            p = os.path.join(d, f"{lp['name']}.lef")
            with open(p, "w") as f:
                f.write(lp["lef_content"])
            ok, out = _run(f"read_lef {techp}\nread_lef {p}\nexit 0\n")
            report["lefs"][lp["name"]] = (ok, None if ok else out)
        for fname, text in verilog_files:
            p = os.path.join(d, fname)
            with open(p, "w") as f:
                f.write(text)
            ok, out = _run(f"read_verilog {p}\nexit 0\n")
            report["verilog"][fname] = (ok, None if ok else out)
    return report


def main() -> int:
    from chia.vlsi.sram_openram.sram_characterize import (
        characterize_top_mems_conf_with_openram,
        export_openram_cache,
        import_openram_cache,
        openram_cache_inventory,
        prefill_openram_cache,
    )
    from chia.vlsi.sram_openram.openram_runner import OpenRAMConfig

    ray.init(address="auto")
    ok = True
    src = [("Top.top.mems.conf", SAMPLE_MEMS_CONF)]
    cfg = OpenRAMConfig(timeout_seconds=3600)

    print("=== 1. characterize (parallel run_openram tasks on the 'openram' resource) ===")
    t0 = time.time()
    char = characterize_top_mems_conf_with_openram(src, cfg, synflop_threshold_bytes=THRESHOLD,
                                                   dispatch_options={"scheduling_strategy": "DEFAULT"})
    print(f"  {len(char.srams)} SRAMs -> {len(char.macros)} macros in {time.time() - t0:.0f} s")
    hdr = f"{'sram':<14}{'macro':<34}{'area_um2':>11}{'clk->dout':>10}{'setup':>7}{'cached':>7}{'elapsed':>8}"
    print(hdr)
    print("-" * len(hdr))
    for s in char.srams:
        r = s.result
        print(f"{s.name:<14}{r.macro_name:<34}{r.area_um2:>11.0f}{r.access_time_ns:>10.3f}{r.setup_time_ns:>7.3f}"
              f"{str(r.from_cache):>7}{r.elapsed_s:>8.0f}")
    expected = {"small_rw", "small_mrw", "small_1r1w"}
    if set(char.sram_names) != expected:
        ok = False
        print(f"  ! expected SRAMs {sorted(expected)}, got {sorted(char.sram_names)}")
    for r in char.macros.values():
        if set(r.lib_contents) != {c.lib_suffix for c in cfg.corners}:
            ok = False
            print(f"  ! {r.macro_name}: corners {sorted(r.lib_contents)}")
        if r.area_um2 <= 0 or r.access_time_ns <= 0:
            ok = False
            print(f"  ! {r.macro_name}: bad summary area={r.area_um2} access={r.access_time_ns}")

    print("\n=== 2. second characterization must be all cache hits ===")
    t0 = time.time()
    char2 = characterize_top_mems_conf_with_openram(src, cfg, synflop_threshold_bytes=THRESHOLD,
                                                    dispatch_options={"scheduling_strategy": "DEFAULT"})
    cached = all(r.from_cache for r in char2.macros.values())
    print(f"  {len(char2.macros)} macros in {time.time() - t0:.0f} s, all from cache: {cached}")
    ok = ok and cached

    print("\n=== 3. OpenROAD validation of libs / LEFs / adapters / stubs ===")
    report = get(validate_collateral.chia_remote(char.sram_libs, char.macro_stub_files))
    print(f"  openroad: {report['openroad']}")
    for section in ("libs", "lefs", "verilog"):
        n_ok = sum(1 for v in report[section].values() if v[0])
        print(f"  {section}: {n_ok}/{len(report[section])} read OK")
        for k, (good, err) in report[section].items():
            if not good:
                ok = False
                print(f"   ! {k}:\n{err}")
    if report["openroad"] is None:
        print("  ! openroad not found on the worker")
        ok = False

    print("\n=== 4. cache management ChiaFunctions ===")
    pre = get(prefill_openram_cache.chia_remote(SAMPLE_MEMS_CONF, cfg, synflop_threshold_bytes=THRESHOLD))
    print(f"  prefill: {len(pre)} macros, all cached: {all(v['from_cache'] for v in pre.values())}")
    inv = get(openram_cache_inventory.chia_remote())
    print(f"  inventory: {len(inv)} entries")
    blob = get(export_openram_cache.chia_remote())
    imported = get(import_openram_cache.chia_remote(blob, overwrite=True))
    print(f"  export: {len(blob)} bytes; re-import (overwrite): {len(imported)} entries")
    ok = ok and all(v["from_cache"] for v in pre.values()) and len(inv) >= len(char.macros) and len(imported) >= len(char.macros)

    print("\n" + ("PASS: OpenRAM macros generated, cached and validated." if ok else "FAIL: see the markers above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
