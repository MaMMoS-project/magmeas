"""
Python module that enables the import and handling of VSM-data acquired on a
Quantum Design PPMS measurement system. Extrinsic magnetic properties can be
calculated from M(H) hysteresis loops and exported to JSON- or CSV-files.
Automatic plotting with matplotlib is also available.
"""

__version__ = "0.5.1"

from magmeas.forc import FORC
from magmeas.mh_major import MH_major
from magmeas.mh_recoil import MH_recoil
from magmeas.mt import MT
from magmeas.sweep import Sweep

__all__ = [
    "Sweep",
    "MH_major",
    "MH_recoil",
    "FORC",
    "MT",
]
