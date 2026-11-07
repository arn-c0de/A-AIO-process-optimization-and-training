"""Command implementations of ``python -m simple_sim <command>`` (plan item P2).

Each module is one command and exposes:

* its module docstring - the first line is the description shown by ``commands``
* ``SUPPORTS_JSON`` - whether it accepts ``--json`` (machine-readable result line)
* ``build_parser(prog=None) -> argparse.ArgumentParser``
* ``main(argv=None, *, prog=None) -> int`` - parse *argv* (default ``sys.argv[1:]``) and run

:mod:`simple_sim.cli` imports them lazily, so listing commands stays torch-free. The scripts in
``scripts/`` are shims that call the same ``main``. Console output (``print``) is allowed here and
in :mod:`simple_sim.cli` only; library modules log.
"""
