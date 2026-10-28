"""Utility classses and functions called within magmas. Not intended for use."""

import mammos_entity as me
import mammos_units as u
import numpy as np


def rextract(superstr, startsub, endsub):
    """
    Extract a substring from a string in reverse order.
    This means the substring between the first occurrence of endsub and from
    there the last occurence of startsub not including either of those two
    will be returned. For example:
        rextract('The quick brown fox', 'quick ', ' fox') will return 'brown'.

    Parameters
    ----------
    superstr : STR
        String from which the substring is to be extracted from.
    startsub : STR
        Substring of superstr after which the substring to be extracted starts.
    endsub : STR
        Substring of superstr before which the substring to be extracted ends.

    Returns
    -------
    STR
        Substring that gets extracted from string.

    """
    endind = superstr.index(endsub)  # starting index
    startind = superstr.rindex(startsub, 0, endind)  # endindex
    return superstr[startind + len(startsub) : endind]


def _MH_unit_processing(unit):
    """Return units and labels used for plotting of MH-derived objects."""
    if isinstance(unit, str | u.Unit | u.CompositeUnit):
        H_unit = u.Unit(unit)
        M_unit = u.Unit(unit)
    else:
        if len(unit) > 2:
            raise ValueError(
                "Length of unit surpasses maximum of two.\n"
                + "Please only pass two values: (H_unit, M_unit)."
            )
        elif len(unit) == 2:
            H_unit = u.Unit(unit[0])
            M_unit = u.Unit(unit[1])
        else:
            H_unit = u.Unit(unit[0])
            M_unit = u.Unit(unit[0])

    if H_unit.is_equivalent(u.T, equivalencies=None):
        H_label = "$\\mu_0 H_{int}$"
    elif H_unit.is_equivalent(u.A / u.m, equivalencies=None):
        H_label = "$H_{int}$"
    else:
        raise ValueError(
            f"Unit {H_unit} is incompatible with magnetic "
            + "field or magnetic flux density units."
        )
    if M_unit.is_equivalent(u.T, equivalencies=None):
        M_label = "J"
    elif M_unit.is_equivalent(u.A / u.m, equivalencies=None):
        M_label = "M"
    else:
        raise ValueError(
            f"Unit {M_unit} is incompatible with magnetisation"
            + " or magnetic polarisation units."
        )
    return H_unit, M_unit, H_label, M_label


def droot(x, y):
    """
    Find root of a discrete dataset of x and y values.

    Parameters
    ----------
    x : ARRAY
        Dataset in horizontal axis, on which root point is located on.
    y : ARRAY
        Dataset in vertical axis.

    Returns
    -------
    FLOAT|ARRAY
        Array of root points. If only one is found, it's returned as float.
    """
    # Boolean array that's true at roots
    r = np.abs(np.diff(np.sign(y))).astype(bool)
    # Calculate roots with linear interception between points
    r = x[:-1][r] + np.diff(x)[r] / (np.abs(y[1:][r] / y[:-1][r]) + 1)
    # Convert array of found roots to float if only one was found
    if np.shape(r) == (1,):
        r = r[0]
    return r


def _arg_processing(sweep, arg):
    """Doe some internal processing to return the values to plot."""
    if isinstance(arg, me.Entity | u.Quantity | np.ndarray | list):
        return arg
    elif isinstance(arg, str):
        if arg in sweep:
            return sweep[arg]
        else:
            arg_key = sweep.get_entity_key(arg)
            if arg_key is None:
                raise AttributeError(
                    f"The given argument was of type string and contained '{arg}'. "
                    + "This could neither be found as a key, nor as an "
                    + "ontology_label. Please check, if the argument was correct."
                )
            elif isinstance(arg_key, list):
                raise AttributeError(
                    f"The ontology_label '{arg}' was found more than once in "
                    + f"the given magmeas object under the keys {arg_key}. "
                    + "Please resolve this ambiguity and try again."
                )
            else:
                return sweep[arg_key]
    else:
        raise TypeError(
            f"The given argument was of type {type(arg)} which is not supported. "
            + "Please only pass keys as strings or objects of type Entity, "
            + "Quantity, ndarray or list to this function."
        )
