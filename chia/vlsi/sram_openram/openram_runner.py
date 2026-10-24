"""Run the OpenRAM SRAM compiler for one SRAM geometry and collect its collateral.

* OpenRAM generates a macro (layout, LEF, Liberty per corner, Verilog
  model) and takes minutes to hours per macro, so :func:`run_openram` is a
  ``@ChiaFunction(resources={"openram": 1})`` meant to be dispatched once per
  distinct geometry, in parallel, and every result goes through the on-disk
  :class:`~chia.vlsi.sram_openram.openram_cache.OpenRAMCache`.
* The collateral is returned **untouched**: the native ``.lib`` (one per
  corner), ``.lef`` and ``.v`` are what downstream synthesis consumes. The
  :class:`OpenRAMResult` also exposes the CACTI-shaped scalar summary (area,
  access time, leakage, ...) parsed from them for reporting.
* Macros are named canonically from their geometry
  (``openram_sky130_1rw_2048x64_m8``), not after the Chipyard SRAM, so identical
  geometries share one macro and one cache entry; a per-SRAM Verilog adapter
  (see :mod:`chia.vlsi.sram_openram.openram_macrocompiler`) bridges the Chipyard
  port names to the macro.

OpenRAM is driven through a Python config file, exactly as its command line
does (``python sram_compiler.py config.py``), under the interpreter that has the
``openram`` package (``CHIA_OPENRAM_PYTHON`` in the chia-openram image).
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, asdict
from typing import Any

from chia.base.ChiaFunction import ChiaFunction
from chia.chipyard.macrocompiler import (  # noqa: F401  (re-exported for callers)
    SRAMSpec,
    parse_mems_conf,
    rename_ports,
    _compute_family,
)
from chia.vlsi.sram_cacti.liberty_gen import LibertyCorner, SKY130_CORNERS
from chia.vlsi.sram_openram.openram_cache import (
    OpenRAMCache,
    cache_key,
    default_cache_dir,
)

ENV_PYTHON = "CHIA_OPENRAM_PYTHON"
ENV_WORK_ROOT = "CHIA_OPENRAM_WORK_ROOT"
DEFAULT_CELL_PREFIX = "openram_"
MAX_PORTS = 2


class OpenRAMError(RuntimeError):
    """OpenRAM could not produce a macro for a spec (never silently fallen back)."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OpenRAMConfig:
    """Knobs for an OpenRAM run. Everything here has a working default for
    sky130; the SRAM geometry itself comes from the :class:`SRAMSpec`.

    Only ``timeout_seconds``, ``openram_python``, ``keep_temp`` and
    ``store_gds`` are excluded from the cache key — every other field changes
    the macro and therefore the key.
    """
    tech_name: str = "sky130"
    #: Liberty corners to characterize (one native ``.lib`` each). Passed to
    #: OpenRAM as ``use_specified_corners`` so the PVT triples match the chia
    #: corner set exactly (default: sky130 ff/ss/tt as in the CACTI flow).
    corners: tuple[LibertyCorner, ...] = SKY130_CORNERS
    #: Analytical (Elmore) timing — seconds. ``False`` means SPICE
    #: characterization: hours per macro and needs ngspice in the image.
    analytical_delay: bool = True
    #: DRC/LVS on the final layout; needs magic+netgen, not in the image.
    check_lvsdrc: bool = False
    #: OpenRAM power routing: ``"ring"`` (default), ``"left"``, ... or ``False``
    #: for none. If the router fails for a geometry (it does for some sky130
    #: shapes) the run is retried with ``False`` and the result says so.
    route_supplies: str | bool = "ring"
    #: Route the signal pins out to the macro perimeter. OpenRAM's escape
    #: router crashes on some sky130 shapes (e.g. masked two-port arrays); the
    #: run is then retried with ``False`` (pins stay on the internal shapes,
    #: which is irrelevant for synthesis) and the result says so.
    perimeter_pins: bool = True
    num_banks: int = 1
    #: ``None`` lets OpenRAM pick the column mux; 1/2/4/8/16 forces it.
    words_per_row: int | None = None
    local_array_size: int = 0
    num_threads: int = 1
    uniquify: bool = True
    #: Largest macro OpenRAM is asked to build. A Chipyard SRAM deeper than
    #: ``max_depth`` (OpenRAM/sky130 refuses more than 512 rows x 8-way mux,
    #: and BOOM's 8192-word L2 bank hits that) or wider than ``max_width`` is
    #: mapped onto several copies of a smaller TILE macro; MacroCompiler
    #: composes the full memory from the tiles (that is its normal job for
    #: vendor macro libraries). ``None`` = no cap.
    max_depth: int | None = 2048
    max_width: int | None = None
    #: Tiling budget: an SRAM whose depth*width (after power-of-two depth
    #: rounding) is at or below this many bits is built as ONE macro even if
    #: it exceeds ``max_width``; only bigger SRAMs are clipped to the caps.
    #: ``None`` applies the caps to every SRAM. Lets the caps describe the
    #: tile for the few huge arrays without splitting every medium one.
    max_tile_bits: int | None = None
    #: Round a non-power-of-two depth UP to the next power of two (BOOM has
    #: 40-word predictor tables; OpenRAM's analytical characterizer fails to
    #: find its probe nets on such arrays, and real compilers only offer
    #: power-of-two depths anyway). MacroCompiler leaves the extra rows unused.
    round_depth_pow2: bool = True
    #: Verbatim ``name = value`` lines appended to the config (values are
    #: rendered with ``repr``), for any OpenRAM option not modelled above.
    extra_options: dict[str, Any] = field(default_factory=dict)
    #: Raw text of an OpenRAM config file (it is Python) inserted *before*
    #: the generated lines, so the spec-derived geometry/ports/output lines
    #: always win.
    config_template: str | None = None
    # --- run-time only (not part of the cache key) ---
    #: Per-macro wall-clock limit. OpenRAM's router needs ~1 h for 1-4 KB
    #: sky130 macros and 3+ h for 2048x256; the cache makes this a one-off.
    timeout_seconds: int = 21600
    openram_python: str | None = None
    keep_temp: bool = False
    store_gds: bool = False

    _NON_KEY_FIELDS = ("timeout_seconds", "openram_python", "keep_temp", "store_gds")

    def key_fields(self) -> dict:
        d = asdict(self)
        for k in self._NON_KEY_FIELDS:
            d.pop(k, None)
        d["corners"] = [asdict(c) for c in self.corners]
        return d


@dataclass
class MacroPort:
    """One OpenRAM macro port as declared in its Verilog model."""
    index: int
    kind: str                 # "rw" | "w" | "r"
    clk: str                  # clk<i>
    csb: str                  # csb<i>  (active low chip select)
    web: str | None           # web<i>  (active low write enable; rw ports only)
    wmask: str | None         # wmask<i> (num_wmasks bits; write-capable ports, masked macros)
    spare_wen: str | None     # spare_wen<i> (spare-column write enable, if spare cols)
    addr: str                 # addr<i>
    din: str | None           # din<i>  (write-capable ports)
    dout: str | None          # dout<i> (read-capable ports)


@dataclass
class OpenRAMResult:
    """One generated macro: untouched OpenRAM collateral plus a scalar summary.

    The scalar fields deliberately share names with ``CACTIResult`` so reports
    and tables written for the CACTI flow work unchanged.
    """
    macro_name: str
    # -- CACTIResult-compatible summary (parsed from the nominal-corner lib/LEF)
    access_time_ns: float        # worst clk->dout delay in the nominal lib (from the FALLING edge, see docs)
    cycle_time_ns: float         # Liberty minimum_period (0 if absent)
    read_energy_nj: float        # not provided by OpenRAM's lib -> 0.0
    leakage_power_mw: float
    height_mm: float
    width_mm: float
    area_um2: float
    full_out: str                # OpenRAM stdout/stderr tail
    # -- extra timing
    setup_time_ns: float
    hold_time_ns: float
    # -- native collateral
    lib_contents: dict[str, str]  # chia corner lib_suffix -> native .lib text
    lef_content: str
    verilog_content: str
    config_py: str                # the config OpenRAM was run with
    # -- macro interface facts (from the Verilog model)
    data_width: int
    addr_width: int
    num_wmasks: int
    ports: list[MacroPort]
    power_pins: tuple[str, str]   # (vdd, gnd) pin names, e.g. ("vccd1", "vssd1")
    # -- provenance
    openram_version: str
    cache_key: str
    from_cache: bool
    elapsed_s: float
    supply_routed: bool
    perimeter_pins: bool = True
    notes: list[str] = field(default_factory=list)

    def to_lib_dict(self) -> dict:
        """Legacy dict shape used by the Hammer node (``cacti_sram_libs``)."""
        return {"name": self.macro_name, "lib_contents": self.lib_contents,
                "lef_content": self.lef_content}


# ---------------------------------------------------------------------------
# Spec validation and naming
# ---------------------------------------------------------------------------

def validate_spec_for_openram(spec: SRAMSpec) -> None:
    """Raise :class:`OpenRAMError` for geometries OpenRAM cannot build.

    sky130 has hand-made 1-port (6T) and 2-port (8T) bitcells; anything with
    three or more ports would silently fall back to OpenRAM's parameterized
    bitcell, which is not what anyone wants from a sky130 macro, so it is an
    error here (no CACTI/analytical fallback either, by design).
    """
    nports = spec.num_rw_ports + spec.num_read_ports + spec.num_write_ports
    if nports == 0:
        raise OpenRAMError(f"{spec.name}: no ports parsed from {spec.ports!r}")
    if nports > MAX_PORTS:
        raise OpenRAMError(
            f"{spec.name}: {nports} ports ({spec.ports}) — OpenRAM/sky130 supports at most "
            f"{MAX_PORTS} (1rw, 1r1w, 1rw1r, 2rw)")
    if spec.num_write_ports + spec.num_rw_ports == 0:
        raise OpenRAMError(f"{spec.name}: read-only memory ({spec.ports}); OpenRAM SRAMs need a write port")
    if spec.num_read_ports + spec.num_rw_ports == 0:
        raise OpenRAMError(f"{spec.name}: write-only memory ({spec.ports}) is not an SRAM")
    if spec.depth < 2 or spec.width < 1:
        raise OpenRAMError(f"{spec.name}: degenerate geometry {spec.depth}x{spec.width}")
    if spec.mask_gran:
        if spec.width % spec.mask_gran != 0:
            raise OpenRAMError(
                f"{spec.name}: width {spec.width} is not a multiple of mask_gran {spec.mask_gran} "
                f"(OpenRAM write_size must divide word_size)")


def tile_spec_for(spec: SRAMSpec, config: OpenRAMConfig) -> SRAMSpec:
    """The geometry OpenRAM actually builds for ``spec``: the same ports and
    mask granularity, depth capped at ``config.max_depth`` and width at
    ``config.max_width`` (a multiple of the mask granularity). Returns ``spec``
    itself when no cap applies. MacroCompiler tiles the SRAM from copies of
    this macro (via the adapter of the same name)."""
    depth, width = spec.depth, spec.width
    if config.round_depth_pow2 and depth & (depth - 1):
        depth = 1 << (depth - 1).bit_length()
    within_budget = config.max_tile_bits is not None and depth * width <= config.max_tile_bits
    # OpenRAM's own hard limit (512 rows x 8-way mux) still applies.
    hard_max_depth = 2048
    if config.max_depth and depth > config.max_depth and not within_budget:
        depth = config.max_depth
    if depth > hard_max_depth:
        depth = hard_max_depth
    if config.max_width and width > config.max_width and not within_budget:
        gran = spec.mask_gran if (spec.mask_gran and any(m for _, m in spec.parsed_ports)) else 1
        w = (config.max_width // gran) * gran
        if w <= 0:
            raise OpenRAMError(f"{spec.name}: max_width {config.max_width} is below the mask granularity {gran}")
        width = w
    if (depth, width) == (spec.depth, spec.width):
        return spec
    return SRAMSpec(name=spec.name, depth=depth, width=width, ports=spec.ports, mask_gran=spec.mask_gran,
                    num_rw_ports=spec.num_rw_ports, num_read_ports=spec.num_read_ports,
                    num_write_ports=spec.num_write_ports, parsed_ports=list(spec.parsed_ports))


def _write_size(spec: SRAMSpec) -> int | None:
    """OpenRAM ``write_size`` for the spec, or None for an unmasked macro.

    A mask granularity equal to the word width (one mask bit for the whole
    word) is not a write mask from OpenRAM's point of view — it would emit no
    ``wmask`` pins — so the macro is built unmasked and the adapter folds the
    single mask bit into the chip select.
    """
    masked = any(m for _, m in spec.parsed_ports)
    if masked and spec.mask_gran and spec.mask_gran < spec.width:
        return spec.mask_gran
    return None


def macro_name_for(spec: SRAMSpec, config: OpenRAMConfig, cell_prefix: str = DEFAULT_CELL_PREFIX) -> str:
    """Canonical macro name from geometry: ``<prefix><tech>_<family>_<depth>x<width>[_m<gran>]``."""
    spec = tile_spec_for(spec, config)
    name = f"{cell_prefix}{config.tech_name}_{_compute_family(spec)}_{spec.depth}x{spec.width}"
    ws = _write_size(spec)
    if ws:
        name += f"_m{ws}"
    return name


# ---------------------------------------------------------------------------
# Config rendering
# ---------------------------------------------------------------------------

def _corner_tuple(c: LibertyCorner) -> tuple[str, float, int]:
    return (c.name.upper(), float(c.voltage), int(c.temperature))


def _openram_corner_suffix(c: LibertyCorner) -> str:
    """The suffix OpenRAM puts on the per-corner .lib file name:
    ``{proc}_{volt}V_{temp}C`` with ``.`` -> ``p`` (``FF_1p95V_-40C``)."""
    proc, volt, temp = _corner_tuple(c)
    return f"{proc}_{volt}V_{temp}C".replace(".", "p")


def _initial_spares(spec: SRAMSpec, config: OpenRAMConfig) -> tuple[int, int]:
    """sky130's bitcell arrays must have an even number of rows and columns
    *including* the replica column / dummy row that a single-port macro adds
    (``array_col_multiple = array_row_multiple = 2``). Every bundled sky130
    1rw macro therefore carries one spare row and one spare column. Start from
    that for 1-port macros; :func:`run_openram` flips a parity on the specific
    OpenRAM error if the guess was wrong (depends on the column mux it picks).
    """
    nports = spec.num_rw_ports + spec.num_read_ports + spec.num_write_ports
    if config.tech_name != "sky130" or nports != 1:
        return 0, 0
    spare_cols = 1 if spec.width % 2 == 0 else 0
    spare_rows = 1 if spec.depth % 2 == 0 else 0
    return spare_rows, spare_cols


def render_openram_config(
    spec: SRAMSpec,
    config: OpenRAMConfig,
    macro_name: str,
    spare_rows: int,
    spare_cols: int,
    route_supplies: str | bool,
    output_path: str | None = None,
    perimeter_pins: bool | None = None,
) -> tuple[str, str]:
    """Return ``(body, full_config)``.

    ``body`` is the deterministic part (everything that defines the macro) and
    is what the cache key hashes; ``full_config`` appends the run-specific
    ``output_path`` line.
    """
    lines: list[str] = []
    if config.config_template:
        lines.append("# ---- user config_template (chia-generated lines below override) ----")
        lines.append(config.config_template.rstrip())
        lines.append("# ---- end config_template ----")
    lines.append("# Generated by chia.vlsi.sram_openram for SRAM spec "
                 f"{spec.name!r}: {spec.depth}x{spec.width} ports={spec.ports} mask_gran={spec.mask_gran}")
    lines.append(f"tech_name = {config.tech_name!r}")
    lines.append(f"word_size = {spec.width}")
    lines.append(f"num_words = {spec.depth}")
    lines.append(f"num_rw_ports = {spec.num_rw_ports}")
    lines.append(f"num_r_ports = {spec.num_read_ports}")
    lines.append(f"num_w_ports = {spec.num_write_ports}")
    ws = _write_size(spec)
    if ws:
        lines.append(f"write_size = {ws}")
    lines.append(f"num_spare_rows = {spare_rows}")
    lines.append(f"num_spare_cols = {spare_cols}")
    lines.append(f"num_banks = {config.num_banks}")
    if config.words_per_row:
        lines.append(f"words_per_row = {config.words_per_row}")
    lines.append(f"local_array_size = {config.local_array_size}")
    lines.append(f"uniquify = {config.uniquify!r}")
    # Flow: full layout (area/LEF need it), no DRC/LVS, analytical timing.
    lines.append("netlist_only = False")
    lines.append(f"check_lvsdrc = {config.check_lvsdrc!r}")
    lines.append(f"analytical_delay = {config.analytical_delay!r}")
    lines.append(f"route_supplies = {route_supplies!r}")
    pp = config.perimeter_pins if perimeter_pins is None else perimeter_pins
    lines.append(f"perimeter_pins = {pp!r}")
    lines.append(f"num_threads = {config.num_threads}")
    # Tool setup: the image provides OpenRAM + PDK on PATH/env; never let
    # OpenRAM install its own miniconda (1.2.x) or demand nix (1.3.x).
    lines.append("use_conda = False")
    lines.append("use_nix = False")
    # Corners: exactly the chia corner set, no auto-generated extras.
    corners = [_corner_tuple(c) for c in config.corners]
    lines.append(f"use_specified_corners = {corners!r}")
    lines.append(f"process_corners = {sorted({p for p, _, _ in corners})!r}")
    lines.append(f"supply_voltages = {sorted({v for _, v, _ in corners})!r}")
    lines.append(f"temperatures = {sorted({t for _, _, t in corners})!r}")
    for k, v in sorted(config.extra_options.items()):
        lines.append(f"{k} = {v!r}")
    lines.append(f"output_name = {macro_name!r}")
    body = "\n".join(lines) + "\n"
    full = body + (f"output_path = {output_path!r}\n" if output_path else "")
    return body, full


# ---------------------------------------------------------------------------
# Locating OpenRAM
# ---------------------------------------------------------------------------

_openram_info_cache: dict[str, tuple[str, str]] = {}


def resolve_openram_python(config: OpenRAMConfig | None = None) -> str:
    py = (config.openram_python if config else None) or os.environ.get(ENV_PYTHON) or sys.executable
    if not (os.path.isfile(py) and os.access(py, os.X_OK)):
        raise OpenRAMError(f"OpenRAM python interpreter not found: {py} (set {ENV_PYTHON})")
    return py


ENV_OPENRAM_HOME = "OPENRAM_HOME"
ENV_OPENRAM_VERSION = "CHIA_OPENRAM_VERSION"


def openram_package_info(python: str) -> tuple[str, str]:
    """``(package_dir, version)`` of the OpenRAM installation to drive.

    ``package_dir`` is the directory holding ``sram_compiler.py``, ``compiler/``
    and ``technology/``. Resolution order:

    1. ``$OPENRAM_HOME`` (its parent dir) — how the chia-openram image installs
       OpenRAM, as a source checkout; the version is ``$CHIA_OPENRAM_VERSION``
       (set by the image: ``<VERSION file>+g<commit>``) or the ``VERSION`` file.
    2. the ``openram`` package importable under ``python`` (a pip install).
    """
    if python in _openram_info_cache:
        return _openram_info_cache[python]
    home = os.environ.get(ENV_OPENRAM_HOME)
    if home and os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(home)), "sram_compiler.py")):
        pkg = os.path.dirname(os.path.abspath(home))
        ver = os.environ.get(ENV_OPENRAM_VERSION)
        if not ver:
            vf = os.path.join(pkg, "VERSION")
            ver = open(vf).read().strip() if os.path.isfile(vf) else "unknown"
        _openram_info_cache[python] = (pkg, ver)
        return pkg, ver
    code = (
        "import importlib.util, importlib.metadata as m, os\n"
        "spec = importlib.util.find_spec('openram')\n"
        "d = spec.submodule_search_locations[0]\n"
        "try:\n    v = m.version('openram')\n"
        "except Exception:\n"
        "    v = open(os.path.join(d, 'VERSION')).read().strip() if os.path.isfile(os.path.join(d, 'VERSION')) else 'unknown'\n"
        "print(d); print(v)\n"
    )
    r = subprocess.run([python, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise OpenRAMError(f"cannot import openram under {python}: {r.stderr.strip()[-500:]}")
    pkg, ver = r.stdout.strip().splitlines()[-2:]
    ver = os.environ.get(ENV_OPENRAM_VERSION) or ver
    _openram_info_cache[python] = (pkg, ver)
    return pkg, ver


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_LEF_SIZE_RE = re.compile(r"^\s*SIZE\s+([\d.]+)\s+BY\s+([\d.]+)\s*;", re.M)
_LIB_AREA_RE = re.compile(r"^\s*area\s*:\s*([\d.eE+\-]+)\s*;", re.M)
_LIB_LEAK_RE = re.compile(r"^\s*cell_leakage_power\s*:\s*([\d.eE+\-]+)\s*;", re.M)
_LIB_LEAK_UNIT_RE = re.compile(r'leakage_power_unit\s*:\s*"1(\w+)"')
_LIB_MINPERIOD_RE = re.compile(r'timing_type\s*:\s*"?minimum_period"?\s*;.*?values\s*\(\s*"([\d.eE+\-]+)"', re.S)
_VALUES_RE = re.compile(r'values\s*\(((?:\s*"[^"]*"\s*,?\s*\\?\s*)+)\)', re.S)


def _table_values(block: str) -> list[float]:
    out: list[float] = []
    for m in _VALUES_RE.finditer(block):
        for num in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1)):
            out.append(float(num))
    return out


def _timing_blocks(lib_text: str, timing_type: str) -> list[str]:
    """Return the text of every ``timing () { ... }`` block with the given type."""
    blocks = []
    for m in re.finditer(r"timing\s*\(\s*\)\s*\{", lib_text):
        start = m.end()
        depth = 1
        i = start
        while i < len(lib_text) and depth:
            c = lib_text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        block = lib_text[start:i]
        if re.search(rf"timing_type\s*:\s*\"?{timing_type}\"?\s*;", block):
            blocks.append(block)
    return blocks


def _max_in_blocks(lib_text: str, timing_type: str, table_names: Sequence[str]) -> float:
    best = None
    for block in _timing_blocks(lib_text, timing_type):
        for tname in table_names:
            for tm in re.finditer(rf"{tname}\s*\([^)]*\)\s*\{{", block):
                # find matching brace for the table
                start = tm.end()
                depth = 1
                i = start
                while i < len(block) and depth:
                    if block[i] == "{":
                        depth += 1
                    elif block[i] == "}":
                        depth -= 1
                    i += 1
                vals = _table_values(block[start:i])
                if vals:
                    v = max(vals)
                    best = v if best is None else max(best, v)
    return best if best is not None else 0.0


_LIB_TYPE_RE = re.compile(r"type\s*\(\s*(\w+)\s*\)\s*\{(.*?)\}", re.S)
_LIB_BUS_RE = re.compile(r"bus\s*\(\s*(\w+)\s*\)\s*\{\s*bus_type\s*:\s*(\w+)", re.S)


def lib_interface(lib_text: str) -> dict:
    """The macro's port interface AS THE LIBERTY DECLARES IT: bus widths from
    the ``type``/``bus`` declarations and the set of pins, plus
    ``memory()``'s address/word width.

    This is what synthesis binds to (Genus resolves the instance to the
    library cell), and it differs from OpenRAM's Verilog model: the model
    carries the spare column (``DATA_WIDTH = word_size + spare cols`` and a
    ``spare_wen`` pin) that the Liberty does not expose.
    """
    type_width = {}
    for name, body in _LIB_TYPE_RE.findall(lib_text):
        m = re.search(r"bit_width\s*:\s*(\d+)", body)
        if m:
            type_width[name] = int(m.group(1))
    bus_width = {bus: type_width.get(t) for bus, t in _LIB_BUS_RE.findall(lib_text)}
    pins = set(re.findall(r"^\s*pin\s*\(\s*(\w+)\s*\)", lib_text, re.M))
    mem = {}
    for key in ("address_width", "word_width"):
        m = re.search(rf"{key}\s*:\s*(\d+)", lib_text)
        if m:
            mem[key] = int(m.group(1))
    return {"bus_width": bus_width, "pins": pins, **mem}


def summarize_native_lib(lib_text: str, lef_text: str) -> dict:
    """CACTI-shaped scalars out of OpenRAM's native nominal-corner Liberty + LEF."""
    m = _LEF_SIZE_RE.search(lef_text)
    if not m:
        raise OpenRAMError("LEF has no SIZE statement")
    width_um, height_um = float(m.group(1)), float(m.group(2))
    am = _LIB_AREA_RE.search(lib_text)
    area_um2 = float(am.group(1)) if am else width_um * height_um
    leak = float(_LIB_LEAK_RE.search(lib_text).group(1)) if _LIB_LEAK_RE.search(lib_text) else 0.0
    unit = _LIB_LEAK_UNIT_RE.search(lib_text)
    scale = {"mW": 1.0, "uW": 1e-3, "nW": 1e-6, "pW": 1e-9, "W": 1e3}.get(unit.group(1) if unit else "mW", 1.0)
    access = max(_max_in_blocks(lib_text, "falling_edge", ("cell_rise", "cell_fall")),
                 _max_in_blocks(lib_text, "rising_edge", ("cell_rise", "cell_fall")))
    setup = _max_in_blocks(lib_text, "setup_rising", ("rise_constraint", "fall_constraint"))
    hold = _max_in_blocks(lib_text, "hold_rising", ("rise_constraint", "fall_constraint"))
    pm = _LIB_MINPERIOD_RE.search(lib_text)
    return dict(
        access_time_ns=access, cycle_time_ns=float(pm.group(1)) if pm else 0.0,
        read_energy_nj=0.0, leakage_power_mw=leak * scale,
        height_mm=height_um / 1000.0, width_mm=width_um / 1000.0, area_um2=area_um2,
        setup_time_ns=setup, hold_time_ns=hold,
    )


_V_PARAM_RE = re.compile(r"parameter\s+(\w+)\s*=\s*(\d+)\s*;")
_V_PORTDECL_RE = re.compile(r"^\s*(input|output|inout)\s*(\[[^\]]+\])?\s*(\w+)\s*;", re.M)


def parse_macro_verilog(v_text: str) -> dict:
    """Extract the module name, parameters, power pins and per-port signal
    names from OpenRAM's Verilog model."""
    mm = re.search(r"^\s*module\s+(\w+)\s*\(", v_text, re.M)
    if not mm:
        raise OpenRAMError("OpenRAM Verilog has no module declaration")
    params = {k: int(v) for k, v in _V_PARAM_RE.findall(v_text)}
    decls = {name: (direction, width) for direction, width, name in _V_PORTDECL_RE.findall(v_text)}
    power = tuple(n for n in decls if decls[n][0] == "inout")
    ports: list[MacroPort] = []
    idx = 0
    while f"clk{idx}" in decls:
        has_din = f"din{idx}" in decls
        has_dout = f"dout{idx}" in decls
        kind = "rw" if (has_din and has_dout) else ("w" if has_din else "r")
        ports.append(MacroPort(
            index=idx, kind=kind, clk=f"clk{idx}", csb=f"csb{idx}",
            web=f"web{idx}" if f"web{idx}" in decls else None,
            wmask=f"wmask{idx}" if f"wmask{idx}" in decls else None,
            spare_wen=f"spare_wen{idx}" if f"spare_wen{idx}" in decls else None,
            addr=f"addr{idx}", din=f"din{idx}" if has_din else None,
            dout=f"dout{idx}" if has_dout else None,
        ))
        idx += 1
    if not ports:
        raise OpenRAMError("OpenRAM Verilog declares no clk0 port")
    return {
        "module": mm.group(1),
        "data_width": params.get("DATA_WIDTH"),
        "addr_width": params.get("ADDR_WIDTH"),
        "num_wmasks": params.get("NUM_WMASKS", 0),
        "ports": ports,
        "power_pins": (power[0], power[1]) if len(power) >= 2 else ("vccd1", "vssd1"),
    }


# ---------------------------------------------------------------------------
# Running OpenRAM
# ---------------------------------------------------------------------------

_COLS_ERR = re.compile(r"Invalid number of cols", re.I)
_ROWS_ERR = re.compile(r"invalid number of rows", re.I)
_SUPPLY_ERR = re.compile(r"supply_router\.py.*Couldn't route", re.I)
_ESCAPE_ERR = re.compile(r"signal_escape_router\.py", re.I)


def _run_once(python: str, pkg_dir: str, config_text: str, work_dir: str,
              timeout_s: int, env_extra: dict | None = None) -> tuple[int, str]:
    os.makedirs(work_dir, exist_ok=True)
    cfg_path = os.path.join(work_dir, "config.py")
    with open(cfg_path, "w") as f:
        f.write(config_text)
    tmp_dir = os.path.join(work_dir, "openram_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("OPENRAM_HOME", os.path.join(pkg_dir, "compiler"))
    env.setdefault("OPENRAM_TECH", os.path.join(pkg_dir, "technology"))
    env["OPENRAM_TMP"] = tmp_dir
    env["PYTHONUNBUFFERED"] = "1"
    if env_extra:
        env.update(env_extra)
    cmd = [python, os.path.join(pkg_dir, "sram_compiler.py"), cfg_path]
    log_path = os.path.join(work_dir, "openram_run.log")
    with open(log_path, "w") as log:
        try:
            r = subprocess.run(cmd, cwd=work_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
                               timeout=timeout_s, text=True)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = -9
            log.write(f"\n[chia] OpenRAM timed out after {timeout_s} s\n")
    with open(log_path, errors="replace") as f:
        out = f.read()
    return rc, out


def _collect_outputs(out_dir: str, macro_name: str, config: OpenRAMConfig) -> dict[str, str | bytes]:
    files: dict[str, str | bytes] = {}
    wanted_text = [f"{macro_name}.v", f"{macro_name}.lef", f"{macro_name}.py", f"{macro_name}.html",
                   f"{macro_name}.log", f"{macro_name}.sp", f"{macro_name}.lvs.sp"]
    for c in config.corners:
        wanted_text.append(f"{macro_name}_{_openram_corner_suffix(c)}.lib")
    for fname in wanted_text:
        p = os.path.join(out_dir, fname)
        if os.path.isfile(p):
            with open(p, errors="replace") as f:
                files[fname] = f.read()
    if config.store_gds:
        p = os.path.join(out_dir, f"{macro_name}.gds")
        if os.path.isfile(p):
            with open(p, "rb") as f:
                files[f"{macro_name}.gds"] = f.read()
    return files


def _result_from_files(spec: SRAMSpec, config: OpenRAMConfig, macro_name: str, files: dict,
                       *, full_out: str, openram_version: str, key: str, from_cache: bool,
                       elapsed_s: float, supply_routed: bool, notes: list[str]) -> OpenRAMResult:
    lib_contents: dict[str, str] = {}
    for c in config.corners:
        fname = f"{macro_name}_{_openram_corner_suffix(c)}.lib"
        if fname not in files:
            raise OpenRAMError(f"{spec.name}: OpenRAM produced no Liberty for corner {c.name} ({fname})")
        lib_contents[c.lib_suffix] = files[fname]
    lef = files.get(f"{macro_name}.lef")
    v = files.get(f"{macro_name}.v")
    if not lef or not v:
        raise OpenRAMError(f"{spec.name}: OpenRAM produced no LEF/Verilog for {macro_name}")
    nominal = next((c for c in config.corners if c.name == "tt"), config.corners[0])
    summary = summarize_native_lib(lib_contents[nominal.lib_suffix], lef)
    vinfo = parse_macro_verilog(v)
    if vinfo["module"] != macro_name:
        notes.append(f"Verilog module name {vinfo['module']} != macro {macro_name}")
    # The interface synthesis sees is the Liberty's, not the Verilog model's:
    # no spare-column data bit, no spare_wen pin (see lib_interface).
    li = lib_interface(lib_contents[nominal.lib_suffix])
    bw = li["bus_width"]
    data_width = bw.get("din0") or bw.get("dout0") or li.get("word_width") or vinfo["data_width"] or spec.width
    addr_width = bw.get("addr0") or li.get("address_width") or vinfo["addr_width"] or max(1, math.ceil(math.log2(max(spec.depth, 2))))
    num_wmasks = next((w for b, w in bw.items() if b.startswith("wmask") and w), None)
    num_wmasks = num_wmasks if num_wmasks is not None else vinfo["num_wmasks"]
    ports = []
    for p in vinfo["ports"]:
        q = MacroPort(**{**p.__dict__})
        q.spare_wen = None if (li["pins"] and q.spare_wen not in li["pins"]) else q.spare_wen
        if q.wmask is not None and q.wmask not in bw and li["pins"] and q.wmask not in li["pins"]:
            q.wmask = None
        ports.append(q)
    if data_width != (vinfo["data_width"] or data_width):
        notes.append(f"Liberty data width {data_width} (Verilog model {vinfo['data_width']}: spare column not in lib)")
    return OpenRAMResult(
        macro_name=macro_name, **summary, full_out=full_out[-4000:],
        lib_contents=lib_contents, lef_content=lef, verilog_content=v,
        config_py=files.get(f"{macro_name}.py", ""),
        data_width=data_width, addr_width=addr_width,
        num_wmasks=num_wmasks, ports=ports, power_pins=vinfo["power_pins"],
        openram_version=openram_version, cache_key=key, from_cache=from_cache,
        elapsed_s=elapsed_s, supply_routed=supply_routed, notes=notes,
    )


@ChiaFunction(resources={"openram": 1})
def run_openram(
    spec: SRAMSpec,
    config: OpenRAMConfig | None = None,
    cache_dir: str | None = None,
    cache_policy: str = "use",
    cell_prefix: str = DEFAULT_CELL_PREFIX,
) -> OpenRAMResult:
    """Generate (or fetch from cache) the OpenRAM macro for one SRAM spec.

    Args:
        spec: Chipyard SRAM spec (from ``parse_mems_conf``).
        config: OpenRAM knobs; defaults to :class:`OpenRAMConfig()`.
        cache_dir: result cache directory on this node; defaults to
            ``$CHIA_OPENRAM_CACHE_DIR`` (set in the chia-openram image). ``""``
            disables caching.
        cache_policy: ``"use"`` (read+write), ``"refresh"`` (ignore existing,
            rewrite), ``"readonly"`` (read, never write).
        cell_prefix: prefix of the canonical macro name.

    Raises:
        OpenRAMError: unsupported geometry, OpenRAM failure or timeout. There
            is deliberately no fallback model.
    """
    config = config or OpenRAMConfig()
    validate_spec_for_openram(spec)
    spec = tile_spec_for(spec, config)
    macro = macro_name_for(spec, config, cell_prefix)
    # Version first, from the environment when the image provides it, so a
    # cache HIT needs no OpenRAM installation at all (e.g. a driver on the
    # head reading an exported/populated cache with cache_policy="readonly").
    version = os.environ.get(ENV_OPENRAM_VERSION)
    python = pkg_dir = None
    if not version:
        python = resolve_openram_python(config)
        pkg_dir, version = openram_package_info(python)

    spare_rows, spare_cols = _initial_spares(spec, config)
    body, _ = render_openram_config(spec, config, macro, spare_rows, spare_cols, config.route_supplies)
    # The key hashes the *intended* config (initial parity guess, requested
    # supply routing); automatic retries are deterministic functions of it.
    key = cache_key(body, version)

    cache = None
    cd = default_cache_dir() if cache_dir is None else cache_dir
    if cd:
        cache = OpenRAMCache(cd)
        if cache_policy in ("use", "readonly"):
            hit = cache.load(key)
            if hit is not None:
                files = hit["files"]
                print(f"  [openram] cache hit {key[:12]} for {spec.name} -> {macro} ({cd})")
                res = _result_from_files(
                    spec, config, macro, files, full_out=hit.get("full_out", ""),
                    openram_version=hit.get("openram_version", version), key=key, from_cache=True,
                    elapsed_s=hit.get("elapsed_s", 0.0), supply_routed=hit.get("supply_routed", True),
                    notes=list(hit.get("notes", [])) + ["from cache"])
                res.perimeter_pins = hit.get("perimeter_pins", True)
                return res

    if cache_policy == "readonly":
        raise OpenRAMError(f"{spec.name}: {macro} (key {key[:12]}) is not in the cache {cd!r} and "
                           f"cache_policy='readonly' forbids running OpenRAM")
    if python is None:
        python = resolve_openram_python(config)
        pkg_dir, _ = openram_package_info(python)
    work_root = os.environ.get(ENV_WORK_ROOT) or None
    work_dir = tempfile.mkdtemp(prefix=f"openram_{macro}_", dir=work_root)
    notes: list[str] = []
    route = config.route_supplies
    perimeter = config.perimeter_pins
    t0 = time.time()
    out = ""
    try:
        for attempt in range(1, 5):
            out_dir = os.path.join(work_dir, f"out{attempt}")
            _, full_cfg = render_openram_config(spec, config, macro, spare_rows, spare_cols, route,
                                                output_path=out_dir, perimeter_pins=perimeter)
            print(f"  [openram] {spec.name}: {macro} attempt {attempt} "
                  f"(spare_rows={spare_rows} spare_cols={spare_cols} route_supplies={route!r} perimeter_pins={perimeter})")
            remaining = max(60, int(config.timeout_seconds - (time.time() - t0)))
            rc, out = _run_once(python, pkg_dir, full_cfg, os.path.join(work_dir, f"run{attempt}"), remaining)
            if rc == 0:
                files = _collect_outputs(out_dir, macro, config)
                elapsed = time.time() - t0
                result = _result_from_files(spec, config, macro, files, full_out=out, openram_version=version,
                                            key=key, from_cache=False, elapsed_s=elapsed,
                                            supply_routed=bool(route), notes=notes)
                result.perimeter_pins = perimeter
                print(f"  [openram] {spec.name}: {macro} OK in {elapsed:.0f} s — "
                      f"{result.width_mm*1000:.1f} x {result.height_mm*1000:.1f} um, "
                      f"area {result.area_um2:.0f} um2, clk->dout {result.access_time_ns:.3f} ns")
                if cache is not None and cache_policy in ("use", "refresh"):
                    meta = {"macro": macro, "spec": {"depth": spec.depth, "width": spec.width, "ports": spec.ports,
                                                     "mask_gran": spec.mask_gran}, "openram_version": version,
                            "elapsed_s": elapsed, "config_body": body, "supply_routed": bool(route),
                            "perimeter_pins": perimeter, "spare_rows": spare_rows, "spare_cols": spare_cols}
                    stored = {"full_out": out[-4000:], "openram_version": version, "elapsed_s": elapsed,
                              "supply_routed": bool(route), "perimeter_pins": perimeter, "notes": notes}
                    cache.store(key, stored, files, meta)
                    print(f"  [openram] cached as {key[:12]} in {cd}")
                return result
            if rc == -9:
                raise OpenRAMError(f"{spec.name}: OpenRAM timed out after {config.timeout_seconds} s for {macro}")
            # Known, deterministic retry cases.
            if _COLS_ERR.search(out):
                spare_cols ^= 1
                notes.append(f"attempt {attempt}: column parity error, retry with num_spare_cols={spare_cols}")
                continue
            if _ROWS_ERR.search(out):
                spare_rows ^= 1
                notes.append(f"attempt {attempt}: row parity error, retry with num_spare_rows={spare_rows}")
                continue
            if _SUPPLY_ERR.search(out) and route:
                notes.append(f"attempt {attempt}: supply router failed with route_supplies={route!r}, retry without power routing")
                route = False
                continue
            if _ESCAPE_ERR.search(out) and perimeter:
                notes.append(f"attempt {attempt}: perimeter pin escape router crashed, retry with perimeter_pins=False")
                perimeter = False
                continue
            break
        err_lines = [l for l in out.splitlines() if l.startswith("ERROR") or "Error" in l][-5:]
        raise OpenRAMError(f"{spec.name}: OpenRAM failed for {macro} (rc={rc}): "
                           + (" | ".join(err_lines) or out[-800:]))
    finally:
        if not config.keep_temp:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            print(f"  [openram] kept work dir {work_dir}")


# ---------------------------------------------------------------------------
# Lib-only re-characterization of cached macros
# ---------------------------------------------------------------------------

_LIB_AREA_LINE_RE = re.compile(r"^\s*area\s*:\s*[^;]+;\s*$", re.M)


def transplant_lib_area(new_lib: str, old_lib: str) -> str:
    """Copy the cell ``area :`` line of ``old_lib`` into ``new_lib``.

    A ``netlist_only`` OpenRAM run has no layout, so its Liberty says
    ``area : 0``; the timing tables are the same as a layout run's. Taking
    the area line from the layout run's Liberty makes the re-characterized
    lib equal to what a full run would have produced.
    """
    om = _LIB_AREA_LINE_RE.search(old_lib)
    nm = _LIB_AREA_LINE_RE.search(new_lib)
    if not om or not nm:
        raise OpenRAMError("Liberty area line not found for transplant")
    return new_lib[:nm.start()] + om.group(0) + new_lib[nm.end():]


def recharacterize_cache_entry(
    cache: OpenRAMCache,
    old_key: str,
    to_version: str,
    timeout_seconds: int = 3600,
    python: str | None = None,
) -> tuple[str, dict]:
    """Regenerate ONLY the Liberty files of a cached macro under a new OpenRAM
    version (e.g. after a patch that changes the analytical timing model),
    reusing the cached layout collateral (LEF/Verilog/SPICE/area).

    OpenRAM is re-run on the entry's own config with ``netlist_only = True``:
    minutes instead of hours, because the layout and routing (what makes
    big macros slow) are skipped, while the netlist and the analytical
    characterization are identical to a full run. The new libs get the cached
    ``area`` line (see :func:`transplant_lib_area`). The result is stored as a
    NEW entry keyed by ``(config body, to_version)``; the old entry is kept.

    Returns ``(new_key, report)``; ``report["tt_identical"]`` says whether the
    nominal-corner lib came out byte-identical to the cached one, which it
    must when the patch only touches other corners (a built-in self-check).
    """
    meta = cache.meta(old_key)
    old = cache.load(old_key)
    if not meta or not old:
        raise OpenRAMError(f"cache entry {old_key} not found")
    body = meta.get("config_body")
    macro = meta.get("macro")
    if not body or not macro:
        raise OpenRAMError(f"cache entry {old_key} has no config_body/macro in meta.json")
    new_key = cache_key(body, to_version)
    if cache.has(new_key):
        return new_key, {"macro": macro, "skipped": "already present"}
    python = python or resolve_openram_python(None)
    pkg_dir, _ = openram_package_info(python)
    work_dir = tempfile.mkdtemp(prefix=f"openram_rechar_{macro}_", dir=os.environ.get(ENV_WORK_ROOT) or None)
    try:
        out_dir = os.path.join(work_dir, "out")
        t0 = time.time()
        # The stored config_body is the INITIAL config; the original run may
        # have flipped the spare row/column parity on OpenRAM's error (the
        # meta records the effective values when available). Replay the same
        # retry logic here.
        body_eff = body
        for dim, key in (("rows", "spare_rows"), ("cols", "spare_cols")):
            if meta.get(key) is not None:
                body_eff = re.sub(rf"^num_spare_{dim} = \d+$", f"num_spare_{dim} = {meta[key]}", body_eff, flags=re.M)
        for attempt in range(1, 4):
            cfg = body_eff.replace("netlist_only = False", "netlist_only = True") + f"output_path = {out_dir!r}\n"
            rc, out = _run_once(python, pkg_dir, cfg, os.path.join(work_dir, f"run{attempt}"), timeout_seconds)
            if rc == 0:
                break
            flipped = None
            for pat, dim in ((_COLS_ERR, "cols"), (_ROWS_ERR, "rows")):
                if pat.search(out):
                    m = re.search(rf"^num_spare_{dim} = (\d+)$", body_eff, re.M)
                    cur = int(m.group(1)) if m else 0
                    body_eff = re.sub(rf"^num_spare_{dim} = \d+$", f"num_spare_{dim} = {cur ^ 1}", body_eff, flags=re.M)
                    flipped = dim
                    break
            if flipped is None:
                break
        if rc != 0:
            err = [l for l in out.splitlines() if l.startswith("ERROR") or "Error" in l][-4:]
            raise OpenRAMError(f"{macro}: netlist-only re-characterization failed (rc={rc}): " + " | ".join(err))
        files = dict(old["files"])
        lib_names = [n for n in files if n.endswith(".lib")]
        report = {"macro": macro, "elapsed_s": round(time.time() - t0), "libs": len(lib_names), "tt_identical": None}
        for name in lib_names:
            p = os.path.join(out_dir, name)
            if not os.path.isfile(p):
                raise OpenRAMError(f"{macro}: re-characterization produced no {name}")
            with open(p, errors="replace") as f:
                new_lib = transplant_lib_area(f.read(), files[name])
            if "_TT_" in name:
                report["tt_identical"] = (new_lib == files[name])
            files[name] = new_lib
        result = {k: v for k, v in old.items() if k != "files"}
        result["openram_version"] = to_version
        result["notes"] = list(result.get("notes", [])) + [f"libs re-characterized (netlist_only) under {to_version} from entry {old_key[:12]}"]
        new_meta = dict(meta)
        new_meta.update({"openram_version": to_version, "recharacterized_from": old_key, "rechar_elapsed_s": report["elapsed_s"]})
        new_meta.pop("key", None); new_meta.pop("stored_at", None)
        cache.store(new_key, result, files, new_meta)
        return new_key, report
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def recharacterize_openram_cache(
    cache_dir: str,
    from_version: str,
    to_version: str,
    timeout_seconds: int = 3600,
    macros: Sequence[str] | None = None,
) -> list[dict]:
    """Re-characterize every entry generated under ``from_version`` into
    ``to_version`` entries (see :func:`recharacterize_cache_entry`)."""
    cache = OpenRAMCache(cache_dir)
    reports = []
    for m in cache.inventory():
        if m.get("openram_version") != from_version:
            continue
        if macros and m.get("macro") not in macros:
            continue
        try:
            key, rep = recharacterize_cache_entry(cache, m["key"], to_version, timeout_seconds)
            rep["new_key"] = key[:12]
        except OpenRAMError as e:
            rep = {"macro": m.get("macro"), "error": str(e)[-300:]}
        print(f"  [openram-rechar] {rep}", flush=True)
        reports.append(rep)
    return reports
