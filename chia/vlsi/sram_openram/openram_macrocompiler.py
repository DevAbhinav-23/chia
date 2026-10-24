"""MacroCompiler glue and Verilog adapters for OpenRAM macros.

Downstream of characterization the OpenRAM flow needs three pieces of
Verilog/JSON collateral per Chipyard SRAM, all generated here:

1. **MDF library** for MacroCompiler (:func:`generate_openram_macrocompiler_lib`):
   one entry per Chipyard SRAM named ``<cell_prefix><sram name>`` with the
   *Chipyard* port names (``RW0_clk``/``RW0_en``/...). This is the generic
   :func:`chia.chipyard.macrocompiler.generate_macrocompiler_lib`, so
   MacroCompiler remaps ``.top.mems.v`` exactly as it does for CACTI.
2. **Adapter module** (:func:`generate_openram_adapters`): the module
   MacroCompiler instantiates, ``<cell_prefix><sram name>``, is a thin Verilog
   wrapper that instantiates the canonical OpenRAM macro and bridges the two
   interfaces: active-high ``*_en``/``*_wmode`` to OpenRAM's active-low
   ``csb``/``web``, zero-extension of the address/data buses (sky130
   single-port macros carry a spare row and a spare column, so the macro's
   ``ADDR_WIDTH``/``DATA_WIDTH`` can exceed the SRAM's), the spare-column
   write enable tied off, and a constant all-ones write mask when the macro is
   masked but the Chipyard port is not.
3. **Blackbox stub** for the macro itself (:func:`generate_openram_macro_stubs`):
   an empty module with the macro's exact port list, so Genus elaborates the
   adapter and binds the instance to the Liberty cell of the same name (the
   same mechanism the CACTI flow uses; OpenRAM's behavioural ``.v`` is NOT
   given to synthesis).

The Liberty and LEF are OpenRAM's own files, untouched.
"""

from __future__ import annotations

import math

from chia.chipyard.macrocompiler import SRAMSpec, generate_macrocompiler_lib
from chia.vlsi.sram_openram.openram_runner import (
    DEFAULT_CELL_PREFIX,
    MacroPort,
    OpenRAMError,
    OpenRAMResult,
)


def generate_openram_macrocompiler_lib(sram_specs: list[SRAMSpec], cell_prefix: str = DEFAULT_CELL_PREFIX) -> str:
    """MDF JSON for MacroCompiler: one ``<cell_prefix><name>`` entry per spec
    (the adapter module name), Chipyard port naming."""
    return generate_macrocompiler_lib(sram_specs, cell_prefix)


def _macro_port_for(spec: SRAMSpec, result: OpenRAMResult, prefix: str) -> MacroPort:
    """Map a Chipyard port prefix (``RW0``/``W1``/``R0``) to the OpenRAM port.

    OpenRAM numbers ports read/write first, then write-only, then read-only
    (``design.py``); Chipyard numbers each kind independently in conf order.
    """
    kind = "rw" if prefix.startswith("RW") else ("w" if prefix.startswith("W") else "r")
    n = int(prefix[2:] if kind == "rw" else prefix[1:])
    candidates = [p for p in result.ports if p.kind == kind]
    if n >= len(candidates):
        raise OpenRAMError(f"{spec.name}: Chipyard port {prefix} has no {kind} port in macro {result.macro_name} "
                           f"({[p.kind + str(p.index) for p in result.ports]})")
    return candidates[n]


def _zext(signal: str, have: int, need: int) -> str:
    """Verilog expression driving a ``need``-bit port from a ``have``-bit signal."""
    if need == have:
        return signal
    if need > have:
        return f"{{{{{need - have}{{1'b0}}}}, {signal}}}"
    raise OpenRAMError(f"signal {signal} ({have} bits) is wider than the macro port ({need} bits)")


def generate_openram_adapter(spec: SRAMSpec, result: OpenRAMResult, cell_prefix: str = DEFAULT_CELL_PREFIX) -> tuple[str, str]:
    """Return ``(filename, verilog)`` of the adapter ``<cell_prefix><spec.name>``."""
    addr_bits = max(1, math.ceil(math.log2(max(spec.depth, 2))))
    mask_bits = math.ceil(spec.width / spec.mask_gran) if spec.mask_gran else 0
    name = f"{cell_prefix}{spec.name}"
    macro = result.macro_name
    dw, aw, nwm = result.data_width, result.addr_width, result.num_wmasks
    if dw < spec.width or aw < addr_bits:
        raise OpenRAMError(f"{spec.name}: macro {macro} is narrower ({dw} bits x 2^{aw}) than the SRAM "
                           f"({spec.width} bits x 2^{addr_bits})")

    port_decls: list[str] = []
    body: list[str] = []
    conns: list[str] = []
    for prefix, masked in spec.parsed_ports:
        mp = _macro_port_for(spec, result, prefix)
        is_rw = prefix.startswith("RW")
        is_r = prefix.startswith("R") and not is_rw
        is_w = prefix.startswith("W")
        port_decls += [f"  input  [{addr_bits - 1}:0] {prefix}_addr",
                       f"  input         {prefix}_clk",
                       f"  input         {prefix}_en"]
        # Chip select: active-low enable. A masked Chipyard port whose single
        # mask bit covers the whole word maps onto an UNMASKED macro (OpenRAM
        # emits no wmask for write_size == word_size), so that bit gates the
        # write through the chip select instead (reads stay enabled).
        whole_word_mask = masked and mask_bits == 1 and mp.wmask is None
        if whole_word_mask and is_rw:
            csb_expr = f"~({prefix}_en & (~{prefix}_wmode | {prefix}_wmask[0]))"
        elif whole_word_mask and is_w:
            csb_expr = f"~({prefix}_en & {prefix}_mask[0])"
        else:
            csb_expr = f"~{prefix}_en"
        conns += [f"    .{mp.clk}({prefix}_clk)",
                  f"    .{mp.csb}({csb_expr})",
                  f"    .{mp.addr}({_zext(prefix + '_addr', addr_bits, aw)})"]
        if is_rw:
            port_decls += [f"  input         {prefix}_wmode",
                           f"  input  [{spec.width - 1}:0] {prefix}_wdata",
                           f"  output [{spec.width - 1}:0] {prefix}_rdata"]
            if masked and mask_bits:
                port_decls.append(f"  input  [{mask_bits - 1}:0] {prefix}_wmask")
            if mp.web is None:
                raise OpenRAMError(f"{spec.name}: macro port {mp.index} of {macro} has no write-enable for {prefix}")
            conns.append(f"    .{mp.web}(~{prefix}_wmode)")
            conns.append(f"    .{mp.din}({_zext(prefix + '_wdata', spec.width, dw)})")
            body.append(f"  wire [{dw - 1}:0] {prefix}_dout_full;")
            conns.append(f"    .{mp.dout}({prefix}_dout_full)")
            body.append(f"  assign {prefix}_rdata = {prefix}_dout_full[{spec.width - 1}:0];")
            wmask_sig = f"{prefix}_wmask" if (masked and mask_bits) else None
        elif is_w:
            port_decls.append(f"  input  [{spec.width - 1}:0] {prefix}_data")
            if masked and mask_bits:
                port_decls.append(f"  input  [{mask_bits - 1}:0] {prefix}_mask")
            conns.append(f"    .{mp.din}({_zext(prefix + '_data', spec.width, dw)})")
            if mp.web is not None:   # a write-only Chipyard port on an rw macro port: always write
                conns.append(f"    .{mp.web}(1'b0)")
            wmask_sig = f"{prefix}_mask" if (masked and mask_bits) else None
        elif is_r:
            port_decls.append(f"  output [{spec.width - 1}:0] {prefix}_data")
            body.append(f"  wire [{dw - 1}:0] {prefix}_dout_full;")
            conns.append(f"    .{mp.dout}({prefix}_dout_full)")
            body.append(f"  assign {prefix}_data = {prefix}_dout_full[{spec.width - 1}:0];")
            if mp.web is not None:   # read-only Chipyard port on an rw macro port: never write
                conns.append(f"    .{mp.web}(1'b1)")
            if mp.din is not None:
                conns.append(f"    .{mp.din}({dw}'b0)")
            wmask_sig = None
        else:
            raise OpenRAMError(f"{spec.name}: unknown port prefix {prefix}")

        if whole_word_mask:
            pass   # folded into the chip select above
        elif mp.wmask is not None:
            if wmask_sig is not None:
                if nwm != mask_bits:
                    raise OpenRAMError(f"{spec.name}: macro {macro} has {nwm} write-mask bits, SRAM port {prefix} has {mask_bits}")
                conns.append(f"    .{mp.wmask}({wmask_sig})")
            else:
                conns.append(f"    .{mp.wmask}({{{nwm}{{1'b1}}}})")
        elif wmask_sig is not None:
            raise OpenRAMError(f"{spec.name}: SRAM port {prefix} is masked but macro {macro} has no write mask")
        if mp.spare_wen is not None:
            conns.append(f"    .{mp.spare_wen}(1'b0)")

    lines = [f"// chia.vlsi.sram_openram adapter: Chipyard SRAM {spec.name} "
             f"({spec.depth}x{spec.width}, ports={spec.ports}) -> OpenRAM macro {macro}",
             f"module {name}(",
             ",\n".join(port_decls),
             ");"]
    lines += body
    lines.append(f"  {macro} mem (")
    lines.append(",\n".join(conns))
    lines.append("  );")
    lines.append("endmodule")
    return f"{name}.v", "\n".join(lines) + "\n"


def generate_openram_adapters(specs: list[SRAMSpec], results: dict[str, OpenRAMResult],
                              cell_prefix: str = DEFAULT_CELL_PREFIX) -> list[tuple[str, str]]:
    """Adapters for every spec; ``results`` maps spec name -> its macro result."""
    return [generate_openram_adapter(s, results[s.name], cell_prefix) for s in specs]


def generate_openram_macro_stub(result: OpenRAMResult) -> tuple[str, str]:
    """Empty blackbox module with the macro's exact ports (power pins omitted,
    as OpenRAM's own model does without ``USE_POWER_PINS``)."""
    dw, aw, nwm = result.data_width, result.addr_width, result.num_wmasks
    decls: list[str] = []
    for p in result.ports:
        decls.append(f"  input         {p.clk}")
        decls.append(f"  input         {p.csb}")
        if p.web:
            decls.append(f"  input         {p.web}")
        if p.wmask:
            decls.append(f"  input  [{nwm - 1}:0] {p.wmask}")
        if p.spare_wen:
            decls.append(f"  input         {p.spare_wen}")
        decls.append(f"  input  [{aw - 1}:0] {p.addr}")
        if p.din:
            decls.append(f"  input  [{dw - 1}:0] {p.din}")
        if p.dout:
            decls.append(f"  output [{dw - 1}:0] {p.dout}")
    text = (f"// OpenRAM macro {result.macro_name}: blackbox for synthesis (timing/area from its Liberty)\n"
            f"module {result.macro_name}(\n" + ",\n".join(decls) + "\n);\nendmodule\n")
    return f"{result.macro_name}.v", text


def generate_openram_macro_stubs(results: dict[str, OpenRAMResult] | list[OpenRAMResult]) -> list[tuple[str, str]]:
    """One stub per distinct macro."""
    seen: dict[str, OpenRAMResult] = {}
    for r in (results.values() if isinstance(results, dict) else results):
        seen.setdefault(r.macro_name, r)
    return [generate_openram_macro_stub(r) for _, r in sorted(seen.items())]
