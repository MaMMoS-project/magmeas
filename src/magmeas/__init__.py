"""
Python module that enables the import and handling of VSM-data acquired on a
Quantum Design PPMS measurement system. Extrinsic magnetic properties can be
calculated from M(H) hysteresis loops and exported to JSON- or CSV-files.
Automatic plotting with matplotlib is also available.
"""

__version__ = "0.4.2"

from magmeas.magmeas import VSM, plot_multiple_VSM

__all__ = [VSM, plot_multiple_VSM]
