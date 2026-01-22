"""
Python module that enables the import and handling of VSM-data acquired on a
Quantum Design PPMS measurement system. Extrinsic magnetic properties can be
calculated from M(H) hysteresis loops and exported to JSON- or CSV-files.
Automatic plotting with matplotlib is also available.
"""

__version__ = "0.4.4"

from magmeas.base import (
    MH,
    MT,
    VSM,
    MH_major,
    mult_properties_to_file,
    plot_multiple_MH_major,
)

__all__ = [VSM, MH, MH_major, MT, mult_properties_to_file, plot_multiple_MH_major]
