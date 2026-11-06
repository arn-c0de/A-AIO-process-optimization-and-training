"""Simple-Sim: Synthetic PCB defect detection system.

The library reports progress and warnings through :mod:`logging` (logger ``simple_sim``
and its children) and never prints. Applications decide where that goes; the CLI scripts
call :func:`simple_sim.cli_utils.configure_cli_logging`.
"""

import logging

__version__ = "1.0.3"

logging.getLogger(__name__).addHandler(logging.NullHandler())
