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
