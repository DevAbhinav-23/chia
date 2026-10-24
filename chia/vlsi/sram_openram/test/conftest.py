"""Shared fixtures for the sram_openram unit tests (hermetic: no OpenRAM run)."""

import os

import pytest

from chia.chipyard.macrocompiler import SRAMSpec, rename_ports
from chia.vlsi.sram_openram.openram_runner import (
    MacroPort,
    OpenRAMResult,
    parse_macro_verilog,
    summarize_native_lib,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _read(name: str) -> str:
    with open(os.path.join(FIXTURES, name)) as f:
        return f.read()


@pytest.fixture
def tiny_v() -> str:
    """OpenRAM 1.2.48 sky130 model of a 16x8 1rw macro with a spare row + column."""
    return _read("tiny_1rw.v")


@pytest.fixture
def tiny_lef() -> str:
    return _read("tiny_1rw.lef")


@pytest.fixture
def tiny_lib() -> str:
    return _read("tiny_1rw_TT_1p8V_25C.lib")


def _build_spec(name="mem", depth=16, width=8, ports="rw", mask_gran=None):
    parsed, num_r, num_w, num_rw = rename_ports(ports)
    return SRAMSpec(name=name, depth=depth, width=width, ports=ports, mask_gran=mask_gran,
                    num_rw_ports=num_rw, num_read_ports=num_r, num_write_ports=num_w,
                    parsed_ports=parsed)


@pytest.fixture
def make_spec():
    return _build_spec


@pytest.fixture
def make_result(tiny_v, tiny_lef, tiny_lib):
    """Factory: an OpenRAMResult built from the tiny fixture, with overrides."""
    def _make(**overrides):
        vinfo = parse_macro_verilog(tiny_v)
        summary = summarize_native_lib(tiny_lib, tiny_lef)
        # Liberty interface (what synthesis binds): 8-bit data, 5-bit addr, no spare_wen.
        ports = [MacroPort(**{**p.__dict__, "spare_wen": None}) for p in vinfo["ports"]]
        kw = dict(
            macro_name=vinfo["module"], **summary, full_out="", lib_contents={"tt_025C_1v80": tiny_lib},
            lef_content=tiny_lef, verilog_content=tiny_v, config_py="",
            data_width=8, addr_width=5, num_wmasks=0,
            ports=ports, power_pins=vinfo["power_pins"], openram_version="1.2.48",
            cache_key="k" * 64, from_cache=False, elapsed_s=1.0, supply_routed=True, notes=[],
        )
        kw.update(overrides)
        return OpenRAMResult(**kw)
    return _make
