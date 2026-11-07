"""Former name of :mod:`simple_sim.rows` (plan A4), kept so existing imports keep working.

``simple_sim.schema`` *is* ``simple_sim.rows`` (the same module object in ``sys.modules``), so
classes, functions and monkeypatches are shared. New code imports :mod:`simple_sim.rows`.
"""

import sys

from simple_sim import rows as _rows

sys.modules[__name__] = _rows
