VLSI
====

API reference for :mod:`chia.vlsi`. These pages are generated from the docstrings in the source, so they stay in sync with the code.

Hammer
------

.. automodule:: chia.vlsi.hammer

SRAM characterization (CACTI)
-----------------------------

CACTI Runner
~~~~~~~~~~~~

.. automodule:: chia.vlsi.sram_cacti.cacti_runner

CACTI Macrocompiler
~~~~~~~~~~~~~~~~~~~

.. automodule:: chia.vlsi.sram_cacti.cacti_macrocompiler

SRAM Characterize
~~~~~~~~~~~~~~~~~

.. automodule:: chia.vlsi.sram_cacti.sram_characterize

Liberty Gen
~~~~~~~~~~~

.. automodule:: chia.vlsi.sram_cacti.liberty_gen

LEF Gen
~~~~~~~

.. automodule:: chia.vlsi.sram_cacti.lef_gen

SRAM macros (OpenRAM)
---------------------

Drop-in alternative to the CACTI characterization: real sky130 macros from the
OpenRAM compiler (native Liberty/LEF/Verilog), cached per worker node.

OpenRAM Runner
~~~~~~~~~~~~~~

.. automodule:: chia.vlsi.sram_openram.openram_runner

OpenRAM Cache
~~~~~~~~~~~~~

.. automodule:: chia.vlsi.sram_openram.openram_cache

OpenRAM Macrocompiler glue
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: chia.vlsi.sram_openram.openram_macrocompiler

SRAM Characterize (OpenRAM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: chia.vlsi.sram_openram.sram_characterize

