"""Unit tests for openram_runner: config rendering, naming, validation, parsers."""

import os

import pytest

from chia.vlsi.sram_cacti.liberty_gen import SKY130_CORNERS
from chia.vlsi.sram_openram.openram_runner import (
    OpenRAMConfig,
    OpenRAMError,
    _initial_spares,
    _openram_corner_suffix,
    macro_name_for,
    parse_macro_verilog,
    render_openram_config,
    summarize_native_lib,
    validate_spec_for_openram,
)
from chia.vlsi.sram_openram.openram_cache import cache_key


class TestValidation:
    @pytest.mark.parametrize("ports", ["rw", "mrw", "read,write", "mwrite,read", "rw,read", "rw,rw"])
    def test_supported(self, make_spec, ports):
        validate_spec_for_openram(make_spec(ports=ports, mask_gran=8 if "m" in ports else None))

    @pytest.mark.parametrize("ports", ["write,write,read", "rw,read,read", "read,read,write"])
    def test_three_ports_rejected(self, make_spec, ports):
        with pytest.raises(OpenRAMError, match="at most 2"):
            validate_spec_for_openram(make_spec(ports=ports))

    def test_read_only_rejected(self, make_spec):
        with pytest.raises(OpenRAMError, match="write port"):
            validate_spec_for_openram(make_spec(ports="read"))

    def test_mask_must_divide_width(self, make_spec):
        with pytest.raises(OpenRAMError, match="mask_gran"):
            validate_spec_for_openram(make_spec(width=20, ports="mrw", mask_gran=8))


class TestNaming:
    def test_canonical_name_is_geometry_only(self, make_spec):
        cfg = OpenRAMConfig()
        a = macro_name_for(make_spec(name="dcache_data", depth=2048, width=64, ports="mrw", mask_gran=8), cfg)
        b = macro_name_for(make_spec(name="other_name", depth=2048, width=64, ports="mrw", mask_gran=8), cfg)
        assert a == b == "openram_sky130_1rw_2048x64_m8"

    def test_unmasked_and_two_port(self, make_spec):
        cfg = OpenRAMConfig()
        assert macro_name_for(make_spec(depth=512, width=88, ports="write,read"), cfg) == "openram_sky130_1r1w_512x88"
        assert macro_name_for(make_spec(depth=64, width=32, ports="rw"), cfg, cell_prefix="x_") == "x_sky130_1rw_64x32"


class TestRender:
    def test_body_contains_geometry_and_tool_flags(self, make_spec):
        spec = make_spec(depth=256, width=32, ports="mrw", mask_gran=8)
        body, full = render_openram_config(spec, OpenRAMConfig(), "m", 1, 1, "ring", output_path="/o")
        for needle in ["word_size = 32", "num_words = 256", "num_rw_ports = 1", "write_size = 8",
                       "num_spare_rows = 1", "num_spare_cols = 1", "use_conda = False", "use_nix = False",
                       "netlist_only = False", "check_lvsdrc = False", "analytical_delay = True",
                       "route_supplies = 'ring'", "output_name = 'm'"]:
            assert needle in body, needle
        assert "output_path" not in body
        assert full.endswith("output_path = '/o'\n")

    def test_corners_are_the_chia_set(self, make_spec):
        body, _ = render_openram_config(make_spec(), OpenRAMConfig(), "m", 0, 0, False)
        assert "use_specified_corners = [('FF', 1.95, -40), ('SS', 1.6, 100), ('TT', 1.8, 25)]" in body

    def test_template_precedes_generated_lines(self, make_spec):
        cfg = OpenRAMConfig(config_template="word_size = 999\nfoo = 1", extra_options={"bar": 2})
        body, _ = render_openram_config(make_spec(width=8), cfg, "m", 0, 0, False)
        assert body.index("word_size = 999") < body.index("word_size = 8")
        assert "foo = 1" in body and "bar = 2" in body

    def test_cache_key_ignores_runtime_fields(self, make_spec):
        spec = make_spec()
        b1, _ = render_openram_config(spec, OpenRAMConfig(timeout_seconds=1), "m", 0, 0, False)
        b2, _ = render_openram_config(spec, OpenRAMConfig(timeout_seconds=99, keep_temp=True), "m", 0, 0, False)
        assert cache_key(b1, "1.2.48") == cache_key(b2, "1.2.48")
        assert cache_key(b1, "1.2.48") != cache_key(b1, "1.2.49")
        b3, _ = render_openram_config(spec, OpenRAMConfig(words_per_row=4), "m", 0, 0, False)
        assert cache_key(b1, "1.2.48") != cache_key(b3, "1.2.48")

    def test_initial_spares_single_port_only(self, make_spec):
        assert _initial_spares(make_spec(depth=64, width=32, ports="rw"), OpenRAMConfig()) == (1, 1)
        assert _initial_spares(make_spec(depth=64, width=32, ports="read,write"), OpenRAMConfig()) == (0, 0)
        assert _initial_spares(make_spec(depth=64, width=32, ports="rw"), OpenRAMConfig(tech_name="freepdk45")) == (0, 0)

    def test_corner_suffix(self):
        assert [_openram_corner_suffix(c) for c in SKY130_CORNERS] == ["FF_1p95V_-40C", "SS_1p6V_100C", "TT_1p8V_25C"]


class TestParsers:
    def test_verilog(self, tiny_v):
        info = parse_macro_verilog(tiny_v)
        assert info["module"] == "t2_tiny_nomask"
        assert (info["data_width"], info["addr_width"], info["num_wmasks"]) == (9, 5, 0)
        assert info["power_pins"] == ("vccd1", "vssd1")
        (p,) = info["ports"]
        assert (p.kind, p.clk, p.csb, p.web, p.wmask, p.spare_wen, p.addr, p.din, p.dout) == \
            ("rw", "clk0", "csb0", "web0", None, "spare_wen0", "addr0", "din0", "dout0")

    def test_lib_lef_summary(self, tiny_lib, tiny_lef):
        s = summarize_native_lib(tiny_lib, tiny_lef)
        assert s["width_mm"] == pytest.approx(0.139495) and s["height_mm"] == pytest.approx(0.15254)
        assert s["area_um2"] == pytest.approx(21278.5673)
        assert s["access_time_ns"] == pytest.approx(0.362)      # max of the clk->dout table
        assert s["setup_time_ns"] == pytest.approx(0.103)
        assert s["hold_time_ns"] == pytest.approx(-0.056)
        assert s["leakage_power_mw"] == pytest.approx(0.000337)
        assert s["read_energy_nj"] == 0.0

    def test_lef_without_size_is_error(self, tiny_lib):
        with pytest.raises(OpenRAMError):
            summarize_native_lib(tiny_lib, "MACRO x\nEND x\n")


class TestPackageInfo:
    def test_openram_home_env_wins(self, tmp_path, monkeypatch):
        from chia.vlsi.sram_openram import openram_runner as orr
        pkg = tmp_path / "OpenRAM"
        (pkg / "compiler").mkdir(parents=True)
        (pkg / "sram_compiler.py").write_text("")
        (pkg / "VERSION").write_text("1.2.49\n")
        monkeypatch.setenv("OPENRAM_HOME", str(pkg / "compiler"))
        monkeypatch.delenv("CHIA_OPENRAM_VERSION", raising=False)
        orr._openram_info_cache.clear()
        assert orr.openram_package_info("/nonexistent/python") == (str(pkg), "1.2.49")
        monkeypatch.setenv("CHIA_OPENRAM_VERSION", "1.2.49+gabc")
        orr._openram_info_cache.clear()
        assert orr.openram_package_info("/nonexistent/python")[1] == "1.2.49+gabc"
        orr._openram_info_cache.clear()


class TestTiling:
    def test_no_cap_returns_same_object(self, make_spec):
        s = make_spec(depth=2048, width=256, ports="rw")
        from chia.vlsi.sram_openram.openram_runner import tile_spec_for
        assert tile_spec_for(s, OpenRAMConfig()) is s

    def test_depth_cap(self, make_spec):
        from chia.vlsi.sram_openram.openram_runner import tile_spec_for
        s = make_spec(name="cc_banks_0_ext", depth=8192, width=256, ports="rw")
        t = tile_spec_for(s, OpenRAMConfig())
        assert (t.depth, t.width, t.name, t.ports) == (2048, 256, "cc_banks_0_ext", "rw")
        assert macro_name_for(s, OpenRAMConfig()) == "openram_sky130_1rw_2048x256"

    def test_width_cap_respects_mask(self, make_spec):
        from chia.vlsi.sram_openram.openram_runner import tile_spec_for
        s = make_spec(depth=128, width=256, ports="mwrite,read", mask_gran=64)
        t = tile_spec_for(s, OpenRAMConfig(max_width=100))
        assert (t.depth, t.width, t.mask_gran) == (128, 64, 64)
        with pytest.raises(OpenRAMError):
            tile_spec_for(s, OpenRAMConfig(max_width=32))

    def test_depth_rounds_up_to_pow2(self, make_spec):
        from chia.vlsi.sram_openram.openram_runner import tile_spec_for
        s = make_spec(name="ghist", depth=40, width=72, ports="write,read")
        t = tile_spec_for(s, OpenRAMConfig())
        assert (t.depth, t.width) == (64, 72)
        assert tile_spec_for(s, OpenRAMConfig(round_depth_pow2=False)) is s
        assert tile_spec_for(make_spec(depth=64), OpenRAMConfig()).depth == 64

    def test_tile_budget_keeps_medium_srams_whole(self, make_spec):
        from chia.vlsi.sram_openram.openram_runner import tile_spec_for
        cfg = OpenRAMConfig(max_depth=512, max_width=128, max_tile_bits=512 * 128)
        t = tile_spec_for(make_spec(name="tag", depth=64, width=176, ports="mrw", mask_gran=22), cfg)
        assert (t.depth, t.width) == (64, 176)                       # 11 Kbit: whole
        t = tile_spec_for(make_spec(name="data", depth=2048, width=8, ports="mwrite,read", mask_gran=2), cfg)
        assert (t.depth, t.width) == (2048, 8)                       # 16 Kbit: whole, within OpenRAM's limit
        t = tile_spec_for(make_spec(name="l2", depth=8192, width=256, ports="rw"), cfg)
        assert (t.depth, t.width) == (512, 128)                      # 2 Mbit: tiled
        t = tile_spec_for(make_spec(name="big", depth=4096, width=8, ports="rw"), OpenRAMConfig(max_depth=None))
        assert t.depth == 2048                                       # hard OpenRAM limit always applies


class TestRecharacterize:
    def test_transplant_area_line(self, tiny_lib):
        from chia.vlsi.sram_openram.openram_runner import transplant_lib_area
        new = tiny_lib.replace("area : 21278.5673;", "area : 0;")
        assert "area : 0;" in new
        out = transplant_lib_area(new, tiny_lib)
        assert out == tiny_lib

    def test_recharacterize_uses_netlist_only_and_new_key(self, tmp_path, tiny_lib, tiny_lef, tiny_v, monkeypatch):
        from chia.vlsi.sram_openram import openram_runner as orr
        from chia.vlsi.sram_openram.openram_cache import OpenRAMCache, cache_key
        cache = OpenRAMCache(str(tmp_path))
        body = "word_size = 8\nnetlist_only = False\noutput_name = 'm'\n"
        old_key = cache_key(body, "v1")
        files = {"m_TT_1p8V_25C.lib": tiny_lib, "m_SS_1p6V_100C.lib": tiny_lib.replace("0.217", "41.0"),
                 "m.lef": tiny_lef, "m.v": tiny_v}
        cache.store(old_key, {"openram_version": "v1", "notes": []}, files, {"macro": "m", "config_body": body, "openram_version": "v1"})
        seen = {}
        def fake_run(python, pkg_dir, cfg_text, work_dir, timeout_s, env_extra=None):
            seen["cfg"] = cfg_text
            out = cfg_text.split("output_path = ")[1].strip().strip("'")
            os.makedirs(out, exist_ok=True)
            for n in ("m_TT_1p8V_25C.lib", "m_SS_1p6V_100C.lib"):
                open(os.path.join(out, n), "w").write(tiny_lib.replace("area : 21278.5673;", "area : 0;"))
            return 0, "ok"
        monkeypatch.setattr(orr, "_run_once", fake_run)
        monkeypatch.setattr(orr, "resolve_openram_python", lambda c: "/bin/python")
        monkeypatch.setattr(orr, "openram_package_info", lambda p: ("/pkg", "v2"))
        new_key, rep = orr.recharacterize_cache_entry(cache, old_key, "v2")
        assert "netlist_only = True" in seen["cfg"] and new_key == cache_key(body, "v2") and new_key != old_key
        assert rep["tt_identical"] is True and rep["libs"] == 2
        new = cache.load(new_key)
        assert new["files"]["m_SS_1p6V_100C.lib"] == tiny_lib          # corrected SS == TT fixture here
        assert new["files"]["m.lef"] == tiny_lef and new["openram_version"] == "v2"
        assert cache.meta(new_key)["recharacterized_from"] == old_key
        assert cache.has(old_key)
        assert orr.recharacterize_cache_entry(cache, old_key, "v2")[1]["skipped"]

    def test_recharacterize_replays_parity_retry(self, tmp_path, tiny_lib, tiny_lef, tiny_v, monkeypatch):
        from chia.vlsi.sram_openram import openram_runner as orr
        from chia.vlsi.sram_openram.openram_cache import OpenRAMCache, cache_key
        cache = OpenRAMCache(str(tmp_path))
        body = "num_spare_rows = 1\nnum_spare_cols = 1\nnetlist_only = False\noutput_name = 'm'\n"
        old_key = cache_key(body, "v1")
        cache.store(old_key, {"openram_version": "v1"}, {"m_TT_1p8V_25C.lib": tiny_lib, "m.lef": tiny_lef, "m.v": tiny_v},
                    {"macro": "m", "config_body": body, "openram_version": "v1"})
        calls = []
        def fake_run(python, pkg_dir, cfg_text, work_dir, timeout_s, env_extra=None):
            calls.append(cfg_text)
            if "num_spare_cols = 1" in cfg_text:
                return 1, "ERROR: file sram_config.py: line 146: Invalid number of cols including rbl(s): 181."
            out = cfg_text.split("output_path = ")[1].strip().strip("'"); os.makedirs(out, exist_ok=True)
            open(os.path.join(out, "m_TT_1p8V_25C.lib"), "w").write(tiny_lib.replace("area : 21278.5673;", "area : 0;"))
            return 0, "ok"
        monkeypatch.setattr(orr, "_run_once", fake_run)
        monkeypatch.setattr(orr, "resolve_openram_python", lambda c: "/bin/python")
        monkeypatch.setattr(orr, "openram_package_info", lambda p: ("/pkg", "v2"))
        key, rep = orr.recharacterize_cache_entry(cache, old_key, "v2")
        assert len(calls) == 2 and "num_spare_cols = 0" in calls[1] and rep["tt_identical"] is True
