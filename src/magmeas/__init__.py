"""
Python module that enables the import and handling of VSM-data acquired on a
Quantum Design PPMS measurement system. Extrinsic magnetic properties can be
calculated from M(H) hysteresis loops and exported to JSON- or CSV-files.
Automatic plotting with matplotlib is also available.
"""

__version__ = "0.5.1"

from magmeas.batch import plot_batch, to_batch, to_hdf5, to_yaml
from magmeas.forc import FORC
from magmeas.mh import MH
from magmeas.mh_major import MH_major
from magmeas.mh_recoil import MH_recoil
from magmeas.mt import MT
from magmeas.sweep import Sweep

__all__ = [
    "Sweep",
    "MH",
    "MH_major",
    "MH_recoil",
    "FORC",
    "MT",
    "plot_batch",
    "to_batch",
    "to_yaml",
    "to_hdf5",
]
