"""Top-level orchestration: generate OpenRAM macros for a design's SRAMs.

Drop-in counterpart of :mod:`chia.vlsi.sram_cacti.sram_characterize`:

* :func:`characterize_top_mems_conf_with_openram` parses ``.top.mems.conf``
  out of ``generated_src_files`` and characterizes every SRAM at or above the
  synflop threshold;
* :func:`characterize_srams_with_openram` does the same for pre-parsed specs;
* both return an :class:`OpenRAMCharacterization` whose ``sram_libs`` /
  ``sram_names`` views feed the existing Hammer node and DB code unchanged.

Differences from the CACTI flow that callers should know:

* Identical geometries share one macro (canonical name), so ``sram_libs`` has
  one entry per **macro**, not per SRAM. ``srams`` is still per SRAM.
* The per-SRAM module MacroCompiler maps to (``openram_<name>``) is a Verilog
  adapter around the macro (``adapter_files``), and the macro itself needs a
  blackbox stub (``macro_stub_files``). Pass ``adapter_files + macro_stub_files``
  where the CACTI flow passes ``generate_macro_stubs(...)``.
* Unsupported SRAMs (3+ ports, read- or write-only) and OpenRAM failures raise
  :class:`OpenRAMError`; there is no analytical/CACTI fallback.
* Macros run in parallel as ``run_openram`` tasks on ``openram`` workers when
  ``parallel=True`` (the default), consulting the per-node result cache.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from chia.base.ChiaFunction import ChiaFunction, get
from chia.chipyard.macrocompiler import SRAMSpec, parse_mems_conf
from chia.vlsi.sram_openram.openram_cache import OpenRAMCache, default_cache_dir
from chia.vlsi.sram_openram.openram_macrocompiler import (
    generate_openram_adapters,
    generate_openram_macro_stubs,
    generate_openram_macrocompiler_lib,
)
from chia.vlsi.sram_openram.openram_runner import (
    DEFAULT_CELL_PREFIX,
    OpenRAMConfig,
    OpenRAMError,
    OpenRAMResult,
    macro_name_for,
    run_openram,
    tile_spec_for,
    validate_spec_for_openram,
)

logger = logging.getLogger(__name__)

# Same default as the CACTI flow: SRAMs smaller than this stay synflops.
DEFAULT_SYNFLOP_THRESHOLD_BYTES = 256


@dataclass
class CharacterizedSRAM:
    """One Chipyard SRAM mapped onto an OpenRAM macro (same shape as the CACTI
    dataclass, plus the macro name and adapter)."""
    name: str
    spec: SRAMSpec            # the Chipyard SRAM as in .top.mems.conf
    tile_spec: SRAMSpec       # the geometry OpenRAM built (== spec unless capped)
    result: OpenRAMResult
    lib_contents: dict[str, str]   # corner lib_suffix -> native OpenRAM Liberty
    lef_content: str
    macro_name: str
    adapter_verilog: str

    def to_lib_dict(self) -> dict:
        return {"name": self.macro_name, "lib_contents": self.lib_contents, "lef_content": self.lef_content}


@dataclass
class OpenRAMCharacterization:
    """Result of :func:`characterize_srams_with_openram`."""
    generated_src_files: list[tuple[str, str]]
    srams: list[CharacterizedSRAM] = field(default_factory=list)
    macros: dict[str, OpenRAMResult] = field(default_factory=dict)   # macro name -> result (unique)
    adapter_files: list[tuple[str, str]] = field(default_factory=list)
    macro_stub_files: list[tuple[str, str]] = field(default_factory=list)

    @property
    def sram_names(self) -> list[str]:
        return [s.name for s in self.srams]

    @property
    def sram_libs(self) -> list[dict]:
        """One Hammer-shaped lib dict per distinct macro."""
        return [r.to_lib_dict() for _, r in sorted(self.macros.items())]

    @property
    def tile_specs(self) -> list[SRAMSpec]:
        """Per-SRAM specs with the TILE geometry: what the MacroCompiler MDF
        must describe (the adapter ``<prefix><name>`` has this geometry;
        MacroCompiler composes the full SRAM from copies of it)."""
        return [s.tile_spec for s in self.srams]

    @property
    def stub_files(self) -> list[tuple[str, str]]:
        """Everything synthesis needs besides the remapped mems.v: adapters + macro stubs."""
        return list(self.adapter_files) + list(self.macro_stub_files)


def _split_specs(sram_specs: list[SRAMSpec], threshold: int) -> tuple[list[SRAMSpec], list[SRAMSpec]]:
    tiny = [s for s in sram_specs if s.size_bytes < threshold]
    large = [s for s in sram_specs if s.size_bytes >= threshold]
    return tiny, large


def _run_macros(
    large: list[SRAMSpec], config: OpenRAMConfig, cache_dir: str | None, cache_policy: str,
    cell_prefix: str, parallel: bool, dispatch_options: dict | None,
) -> dict[str, OpenRAMResult]:
    """Generate one macro per distinct canonical name; returns macro name -> result."""
    by_macro: dict[str, SRAMSpec] = {}
    for s in large:
        by_macro.setdefault(macro_name_for(s, config, cell_prefix), s)
    print(f"  [openram] {len(large)} SRAMs -> {len(by_macro)} distinct macros"
          f" ({'parallel' if parallel else 'serial'}; cache_dir={cache_dir or default_cache_dir()!r}, policy={cache_policy})")
    results: dict[str, OpenRAMResult] = {}
    if parallel:
        handle = run_openram.options(**dispatch_options) if dispatch_options else run_openram
        refs = {m: handle.chia_remote(s, config, cache_dir, cache_policy, cell_prefix) for m, s in by_macro.items()}
        errors = []
        for m, ref in refs.items():
            try:
                results[m] = get(ref)
            except Exception as e:  # collect every failure, then raise once
                errors.append(f"{m}: {e}")
        if errors:
            raise OpenRAMError("OpenRAM failed for %d macro(s):\n  " % len(errors) + "\n  ".join(errors))
    else:
        for m, s in by_macro.items():
            results[m] = run_openram(s, config, cache_dir, cache_policy, cell_prefix)
    return results


def characterize_srams_with_openram(
    sram_specs: list[SRAMSpec],
    generated_src_files: list[tuple[str, str]],
    config: OpenRAMConfig | None = None,
    cache_dir: str | None = None,
    cache_policy: str = "use",
    synflop_threshold_bytes: int = DEFAULT_SYNFLOP_THRESHOLD_BYTES,
    cell_prefix: str = DEFAULT_CELL_PREFIX,
    parallel: bool = True,
    dispatch_options: dict | None = None,
) -> OpenRAMCharacterization:
    """Generate OpenRAM macros for ``sram_specs`` and the Verilog glue to use them.

    Args:
        sram_specs: parsed Chipyard SRAM specs.
        generated_src_files: passed through unchanged into the result.
        config: OpenRAM knobs (see :class:`OpenRAMConfig`).
        cache_dir: per-node result cache; ``None`` = the image default
            (``$CHIA_OPENRAM_CACHE_DIR``), ``""`` = no cache.
        cache_policy: ``"use"`` | ``"refresh"`` | ``"readonly"``.
        synflop_threshold_bytes: SRAMs smaller than this stay synflops.
        cell_prefix: prefix of both the adapter modules (``<prefix><sram>``,
            what MacroCompiler maps to) and the canonical macro names.
        parallel: dispatch ``run_openram`` per macro via ``.chia_remote``
            (requires a running Ray cluster with ``openram`` workers); False
            runs them serially in this process.
        dispatch_options: extra ``.options(...)`` for the dispatched tasks,
            e.g. ``{"scheduling_strategy": "DEFAULT"}`` when calling from
            inside a placement group.

    Raises:
        OpenRAMError: on any unsupported SRAM or OpenRAM failure (no fallback).
    """
    config = config or OpenRAMConfig()
    tiny, large = _split_specs(sram_specs, synflop_threshold_bytes)
    print(f"  [openram] Tiny SRAMs (synflop, < {synflop_threshold_bytes}B): {len(tiny)}")
    for s in tiny:
        print(f"    - {s.name}: {s.depth}x{s.width} {s.ports} ({s.size_bytes}B)")
    print(f"  [openram] Large SRAMs (OpenRAM macros): {len(large)}")
    for s in large:
        print(f"    - {s.name}: {s.depth}x{s.width} {s.ports} ({s.size_bytes}B) -> {macro_name_for(s, config, cell_prefix)}")

    # Fail fast, before any multi-minute OpenRAM run, on geometries we refuse.
    problems = []
    for s in large:
        try:
            validate_spec_for_openram(s)
        except OpenRAMError as e:
            problems.append(str(e))
    if problems:
        raise OpenRAMError("unsupported SRAM(s) for OpenRAM:\n  " + "\n  ".join(problems))

    if not large:
        return OpenRAMCharacterization(generated_src_files=generated_src_files)

    tiles = [tile_spec_for(s, config) for s in large]
    for s, t in zip(large, tiles):
        if t is not s:
            print(f"    - {s.name}: {s.depth}x{s.width} tiled as "
                  f"{-(-s.depth // t.depth)}x{-(-s.width // t.width)} copies of {t.depth}x{t.width}")
    macros = _run_macros(tiles, config, cache_dir, cache_policy, cell_prefix, parallel, dispatch_options)
    per_spec = {t.name: macros[macro_name_for(t, config, cell_prefix)] for t in tiles}
    adapters = generate_openram_adapters(tiles, per_spec, cell_prefix)
    stubs = generate_openram_macro_stubs(macros)

    srams = []
    for s, t, (_, adapter_v) in zip(large, tiles, adapters):
        r = per_spec[s.name]
        srams.append(CharacterizedSRAM(
            name=s.name, spec=s, tile_spec=t, result=r, lib_contents=r.lib_contents, lef_content=r.lef_content,
            macro_name=r.macro_name, adapter_verilog=adapter_v))
    n_cached = sum(1 for r in macros.values() if r.from_cache)
    print(f"  [openram] {len(macros)} macros ready ({n_cached} from cache), "
          f"{len(adapters)} adapters, {len(stubs)} stubs, {len(config.corners)} corners each")
    return OpenRAMCharacterization(generated_src_files=generated_src_files, srams=srams, macros=macros,
                                   adapter_files=adapters, macro_stub_files=stubs)


def characterize_top_mems_conf_with_openram(
    generated_src_files: list[tuple[str, str]],
    config: OpenRAMConfig | None = None,
    cache_dir: str | None = None,
    cache_policy: str = "use",
    synflop_threshold_bytes: int = DEFAULT_SYNFLOP_THRESHOLD_BYTES,
    cell_prefix: str = DEFAULT_CELL_PREFIX,
    parallel: bool = True,
    dispatch_options: dict | None = None,
) -> OpenRAMCharacterization:
    """Find ``.top.mems.conf`` in ``generated_src_files`` and characterize its SRAMs
    (see :func:`characterize_srams_with_openram`). No conf -> empty result."""
    mems_conf_content = None
    mems_conf_name = None
    for fname, content in generated_src_files:
        if fname.endswith(".top.mems.conf"):
            mems_conf_content, mems_conf_name = content, fname
            break
    if mems_conf_content is None:
        print("  [openram] No .top.mems.conf found — skipping SRAM characterization")
        return OpenRAMCharacterization(generated_src_files=generated_src_files)
    specs = parse_mems_conf(mems_conf_content)
    print(f"  [openram] Parsed {len(specs)} SRAMs from {mems_conf_name}")
    return characterize_srams_with_openram(
        specs, generated_src_files, config, cache_dir, cache_policy, synflop_threshold_bytes,
        cell_prefix, parallel, dispatch_options)


# ---------------------------------------------------------------------------
# Cache management ChiaFunctions (run on an openram worker, where the cache lives)
# ---------------------------------------------------------------------------

@ChiaFunction(resources={"openram": 1})
def prefill_openram_cache(
    sram_specs_or_mems_conf: list[SRAMSpec] | str,
    config: OpenRAMConfig | None = None,
    cache_dir: str | None = None,
    synflop_threshold_bytes: int = DEFAULT_SYNFLOP_THRESHOLD_BYTES,
    cell_prefix: str = DEFAULT_CELL_PREFIX,
) -> dict[str, dict]:
    """Populate this node's cache for every distinct macro the specs need.

    Runs the macros *serially on this worker* (it already holds one
    ``openram`` slot). Dispatch several of these, one per spec subset, to use
    more workers. Returns ``macro name -> {key, from_cache, elapsed_s, area_um2}``.
    """
    specs = (parse_mems_conf(sram_specs_or_mems_conf) if isinstance(sram_specs_or_mems_conf, str)
             else list(sram_specs_or_mems_conf))
    config = config or OpenRAMConfig()
    _, large = _split_specs(specs, synflop_threshold_bytes)
    for s in large:
        validate_spec_for_openram(s)
    by_macro: dict[str, SRAMSpec] = {}
    for s in large:
        by_macro.setdefault(macro_name_for(s, config, cell_prefix), tile_spec_for(s, config))
    out = {}
    for m, s in sorted(by_macro.items()):
        r = run_openram(s, config, cache_dir, "use", cell_prefix)
        out[m] = {"key": r.cache_key, "from_cache": r.from_cache, "elapsed_s": r.elapsed_s,
                  "area_um2": r.area_um2, "access_time_ns": r.access_time_ns}
    return out


@ChiaFunction(resources={"openram": 0.01})
def openram_cache_inventory(cache_dir: str | None = None) -> list[dict]:
    """Metadata of every entry in this node's cache."""
    cd = default_cache_dir() if cache_dir is None else cache_dir
    if not cd:
        return []
    return OpenRAMCache(cd).inventory()


@ChiaFunction(resources={"openram": 0.01})
def export_openram_cache(cache_dir: str | None = None, keys: list[str] | None = None) -> bytes:
    """Tar+gz of this node's cache (or the given keys), e.g. to persist it on the head."""
    cd = default_cache_dir() if cache_dir is None else cache_dir
    if not cd:
        raise OpenRAMError("no cache_dir (set CHIA_OPENRAM_CACHE_DIR or pass cache_dir)")
    return OpenRAMCache(cd).export_tar(keys)


@ChiaFunction(resources={"openram": 0.01})
def import_openram_cache(blob: bytes, cache_dir: str | None = None, overwrite: bool = False) -> list[str]:
    """Unpack a tarball from :func:`export_openram_cache` into this node's cache."""
    cd = default_cache_dir() if cache_dir is None else cache_dir
    if not cd:
        raise OpenRAMError("no cache_dir (set CHIA_OPENRAM_CACHE_DIR or pass cache_dir)")
    return OpenRAMCache(cd).import_tar(blob, overwrite=overwrite)
