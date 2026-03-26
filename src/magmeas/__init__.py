"""
Python module that enables the import and handling of VSM-data acquired on a
Quantum Design PPMS measurement system. Extrinsic magnetic properties can be
calculated from M(H) hysteresis loops and exported to JSON- or CSV-files.
Automatic plotting with matplotlib is also available.
"""

__version__ = "0.5.0"

from magmeas.base import (
    FORC,
    MH,
    MT,
    VSM,
    MH_major,
    MH_recoil,
    plot_batch,
    to_batch,
    to_hdf5,
    to_yaml,
)

__all__ = [
    VSM,
    MH,
    MH_major,
    MH_recoil,
    FORC,
    MT,
    plot_batch,
    to_batch,
    to_yaml,
    to_hdf5,
]
