"""
Python module that enables the import and handling of VSM-data acquired on a
Quantum Design PPMS measurement system. Extrinsic magnetic properties can be
calculated from M(H) hysteresis loops and exported to JSON- or CSV-files.
Automatic plotting with matplotlib is also available.
"""

__version__ = "0.3.0"

from magmeas.magmeas import (
    VSM,
    mult_properties_to_json,
    mult_properties_to_txt,
    plot_multiple_VSM,
)

__all__ = [VSM, plot_multiple_VSM,
           mult_properties_to_json, mult_properties_to_txt]
