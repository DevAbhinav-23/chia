"""Unit tests for the on-disk OpenRAM result cache."""

from chia.vlsi.sram_openram.openram_cache import OpenRAMCache, cache_key


def _fill(cache, key, macro="m"):
    files = {f"{macro}.v": "module m; endmodule\n", f"{macro}.lef": "SIZE 1 BY 2 ;\n",
             f"{macro}_TT_1p8V_25C.lib": "library(m){}\n", f"{macro}.gds": b"\x00\x01"}
    cache.store(key, {"elapsed_s": 3.0, "openram_version": "1.2.48"}, files, {"macro": macro})


def test_roundtrip(tmp_path):
    c = OpenRAMCache(str(tmp_path / "cache"))
    k = cache_key("body", "1.2.48")
    assert not c.has(k) and c.load(k) is None
    _fill(c, k)
    assert c.has(k)
    got = c.load(k)
    assert got["elapsed_s"] == 3.0
    assert set(got["files"]) == {"m.v", "m.lef", "m_TT_1p8V_25C.lib"}   # gds not inlined
    assert c.meta(k)["macro"] == "m" and c.meta(k)["key"] == k
    assert c.keys() == [k]


def test_store_is_atomic_overwrite(tmp_path):
    c = OpenRAMCache(str(tmp_path))
    k = cache_key("b", "v")
    _fill(c, k, macro="a")
    _fill(c, k, macro="b")
    assert set(c.load(k)["files"]) == {"b.v", "b.lef", "b_TT_1p8V_25C.lib"}
    assert not [d for d in tmp_path.iterdir() if d.name.startswith(".")]   # no temp dirs left


def test_export_import(tmp_path):
    src = OpenRAMCache(str(tmp_path / "src"))
    k1, k2 = cache_key("1", "v"), cache_key("2", "v")
    _fill(src, k1, "m1")
    _fill(src, k2, "m2")
    blob = src.export_tar()
    dst = OpenRAMCache(str(tmp_path / "dst"))
    assert sorted(dst.import_tar(blob)) == sorted([k1, k2])
    assert dst.load(k2)["files"]["m2.v"] == "module m; endmodule\n"
    assert dst.import_tar(blob) == []                    # existing entries kept
    assert sorted(dst.import_tar(blob, overwrite=True)) == sorted([k1, k2])
    assert len(src.export_tar([k1])) < len(blob)
