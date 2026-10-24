#!/usr/bin/env python3
"""Patch OpenRAM's net_spice.connect_pin to O(1) duplicate detection.

Upstream (stable branch, 2026-08) checks ``if pin in self.pins`` — a linear
scan of the net's pin list through pin_spice.__eq__ — for EVERY pin connected
to a net. The bitcell array connects every cell's vdd/gnd/wl/bl pins to the
same few nets, so building the netlist of a 2048x256 macro (524K cells) is
quadratic and never finishes (BOOM's L2 bank). This keeps the semantics (an
identity-based duplicate check, warning on a repeat) with a set of ids that is
rebuilt whenever it is missing or stale (e.g. after a deepcopy of the net).

Usage: net_spice_connect_pin.py <OpenRAM root>   (idempotent)
"""
import sys, os
p = os.path.join(sys.argv[1], "compiler", "base", "hierarchy_spice.py")
s = open(p).read()
old = '''    def connect_pin(self, pin):
        debug.check(isinstance(pin, pin_spice), "pin must be a pin_spice object")
        if pin in self.pins:
            debug.warning("pin {} was already connected to net {} ... why was it connected again?".format(pin.name, self.name))
        else:
            self.pins.append(pin)
'''
new = '''    def connect_pin(self, pin):
        debug.check(isinstance(pin, pin_spice), "pin must be a pin_spice object")
        # chia patch: identity set instead of `pin in self.pins` (a linear scan
        # that made netlist construction quadratic on vdd/gnd/wl/bl nets of
        # large bitcell arrays). Rebuilt when missing or stale (deepcopy).
        ids = self.__dict__.get("_pin_ids")
        if ids is None or len(ids) != len(self.pins) or (self.pins and id(self.pins[-1]) not in ids):
            ids = self._pin_ids = {id(q) for q in self.pins}
        if id(pin) in ids:
            debug.warning("pin {} was already connected to net {} ... why was it connected again?".format(pin.name, self.name))
        else:
            self.pins.append(pin)
            ids.add(id(pin))
'''
if new in s:
    print("already patched"); sys.exit(0)
assert s.count(old) == 1, "connect_pin not found in expected form"
open(p, "w").write(s.replace(old, new)); print("patched", p)
