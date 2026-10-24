#!/usr/bin/env python3
"""Patch OpenRAM's analytical temperature delay factor (hierarchy_spice.py).

Upstream computes the thermal voltage as ``0.008625 * T_celsius`` (volts),
i.e. ~0.86 V at 100 C instead of kT/q ~ 0.032 V, so the threshold shift
``2 * (Vt(T) - Vt(nom))`` reaches ~1.3 V and the delay factor
``(Vdd - Vt_nom) / (Vdd - Vt(T))`` becomes ~80x at 100 C (and the SS corner
Liberty of every macro gets clk->dout of 40-56 ns while TT is 0.4-0.6 ns).
Upstream never sees it because its sky130 flow characterizes the nominal
corner only. Use the physical thermal voltage k*T/q with T in kelvin; the
factor then stays within a few percent, as the comment in the code intends.

Usage: temp_delay_factor.py <OpenRAM root>   (idempotent)
"""
import sys, os
p = os.path.join(sys.argv[1], "compiler", "base", "hierarchy_spice.py")
s = open(p).read()
old = '''        thermal_voltage_nom = 0.008625 * tech.spice["nom_temperature"]
        thermal_voltage = 0.008625 * temp
'''
new = '''        # chia patch: kT/q with T in kelvin (upstream used 0.008625 * T_celsius,
        # ~27x too large, giving an ~80x delay factor at 100 C).
        thermal_voltage_nom = 8.617333e-5 * (tech.spice["nom_temperature"] + 273.15)
        thermal_voltage = 8.617333e-5 * (temp + 273.15)
'''
if new in s:
    print("already patched"); sys.exit(0)
assert s.count(old) == 1, "temperature factor not found in expected form"
open(p, "w").write(s.replace(old, new)); print("patched", p)
