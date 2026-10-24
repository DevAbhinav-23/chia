"""Unit tests for the MDF library, Verilog adapters and macro stubs."""

import json
import re

import pytest

from chia.vlsi.sram_openram.openram_macrocompiler import (
    generate_openram_adapter,
    generate_openram_macro_stubs,
    generate_openram_macrocompiler_lib,
)
from chia.vlsi.sram_openram.openram_runner import MacroPort, OpenRAMError


def test_mdf_uses_adapter_names_and_chipyard_ports(make_spec):
    specs = [make_spec(name="a_ext", depth=16, width=8, ports="mrw", mask_gran=8)]
    entries = json.loads(generate_openram_macrocompiler_lib(specs))
    assert entries[0]["name"] == "openram_a_ext"
    assert entries[0]["ports"][0]["clock port name"] == "RW0_clk"
    assert entries[0]["ports"][0]["mask port name"] == "RW0_wmask"


def test_adapter_1rw_zero_extends_and_inverts(make_spec, make_result):
    # tiny fixture macro, Liberty interface: 8-bit data, 5-bit addr (16 words + spare row), no spare_wen
    spec = make_spec(name="a_ext", depth=16, width=8, ports="rw")
    fname, v = generate_openram_adapter(spec, make_result())
    assert fname == "openram_a_ext.v"
    assert "module openram_a_ext(" in v
    assert ".csb0(~RW0_en)" in v and ".web0(~RW0_wmode)" in v
    assert ".addr0({{1{1'b0}}, RW0_addr})" in v
    assert ".din0(RW0_wdata)" in v
    assert "assign RW0_rdata = RW0_dout_full[7:0];" in v
    assert "spare_wen" not in v
    assert "t2_tiny_nomask mem (" in v
    assert "wmask" not in v


def test_lib_interface_drops_spare_column(tiny_lib):
    from chia.vlsi.sram_openram.openram_runner import lib_interface
    li = lib_interface(tiny_lib)
    assert li["bus_width"] == {"din0": 8, "dout0": 8, "addr0": 5}
    assert (li["word_width"], li["address_width"]) == (8, 5)
    assert "spare_wen0" not in li["pins"] and "csb0" in li["pins"]


def test_adapter_masked_port_needs_masked_macro(make_spec, make_result):
    # two mask bits (16-bit word, 8-bit granularity) cannot fold into csb
    spec = make_spec(name="a_ext", depth=16, width=8, ports="mrw", mask_gran=4)
    with pytest.raises(OpenRAMError, match="no write mask"):
        generate_openram_adapter(spec, make_result())


def test_adapter_1r1w_on_two_port_macro(make_spec, make_result):
    ports = [
        MacroPort(0, "w", "clk0", "csb0", None, "wmask0", None, "addr0", "din0", None),
        MacroPort(1, "r", "clk1", "csb1", None, None, None, "addr1", None, "dout1"),
    ]
    res = make_result(macro_name="openram_sky130_1r1w_512x88_m8", ports=ports, data_width=88, addr_width=9, num_wmasks=11)
    spec = make_spec(name="tag_ext", depth=512, width=88, ports="mwrite,read", mask_gran=8)
    _, v = generate_openram_adapter(spec, res)
    assert ".clk0(W0_clk)" in v and ".csb0(~W0_en)" in v and ".din0(W0_data)" in v and ".wmask0(W0_mask)" in v
    assert ".clk1(R0_clk)" in v and ".csb1(~R0_en)" in v and ".addr1(R0_addr)" in v
    assert "assign R0_data = R0_dout_full[87:0];" in v
    assert "web" not in v


def test_adapter_unmasked_port_on_masked_macro_ties_mask_high(make_spec, make_result):
    ports = [MacroPort(0, "rw", "clk0", "csb0", "web0", "wmask0", None, "addr0", "din0", "dout0")]
    res = make_result(ports=ports, num_wmasks=4, data_width=32, addr_width=6)
    spec = make_spec(name="x", depth=64, width=32, ports="rw")
    _, v = generate_openram_adapter(spec, res)
    assert ".wmask0({4{1'b1}})" in v


def test_macro_stub_declares_every_port(make_result):
    (fname, text), = generate_openram_macro_stubs([make_result(), make_result()])   # dedupes
    assert fname == "t2_tiny_nomask.v"
    for p in ["clk0", "csb0", "web0", "addr0", "din0", "dout0"]:
        assert re.search(rf"\b{p}\b", text)
    assert "spare_wen" not in text
    assert "input  [7:0] din0" in text and "output [7:0] dout0" in text and "input  [4:0] addr0" in text
    assert text.rstrip().endswith("endmodule")


def test_whole_word_mask_folds_into_chip_select(make_spec, make_result):
    # mask_gran == width: OpenRAM builds an unmasked macro; the one mask bit gates csb.
    from chia.vlsi.sram_openram.openram_runner import OpenRAMConfig, macro_name_for
    spec = make_spec(name="x", depth=16, width=8, ports="mrw", mask_gran=8)
    assert macro_name_for(spec, OpenRAMConfig()) == "openram_sky130_1rw_16x8"
    _, v = generate_openram_adapter(spec, make_result())
    assert ".csb0(~(RW0_en & (~RW0_wmode | RW0_wmask[0])))" in v
    ports = [MacroPort(0, "w", "clk0", "csb0", None, None, None, "addr0", "din0", None),
             MacroPort(1, "r", "clk1", "csb1", None, None, None, "addr1", None, "dout1")]
    res = make_result(macro_name="openram_sky130_1r1w_64x8", ports=ports, data_width=8, addr_width=6, num_wmasks=0)
    spec = make_spec(name="y", depth=64, width=8, ports="mwrite,read", mask_gran=8)
    _, v = generate_openram_adapter(spec, res)
    assert ".csb0(~(W0_en & W0_mask[0]))" in v and ".csb1(~R0_en)" in v
