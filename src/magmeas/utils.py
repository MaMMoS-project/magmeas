"""Utility functions called within magmas. They will not be used by the user."""

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
    r = np.array([]) * x.unit
    # scan over whole range of values to find the two points where the
    # y-axis is crossed
    for i in range(len(x) - 1):
        # y values on left and right side of root will only be negative if
        # their product is negative
        if y[i] * y[i + 1] <= 0:
            # dataset between two points is assumed to be linear,
            # calculate linear equation to get exact interception point
            # with x-axis
            m = (y[i + 1] - y[i]) / (x[i + 1] - x[i])  # slope
            n = y[i] - m * x[i]  # y-intercept
            x0 = -n / m  # x-intercept
            # append x-intercepts as root to array
            r = np.append(r, x0)
    # Convert array of found roots to float if only one was found
    if np.shape(r) == (1,):
        r = r[0]
    return r


def diff(a, b):
    """
    Calculate the difference between value a and value b relative to
    value a in percent.

    Parameters
    ----------
    a : INT | FLOAT
        Numerical value.
    b : INT | FLOAT
        Numerical value.

    Returns
    -------
    FLOAT
        Difference of Value a and b in percent.
    """
    return (b - a) / a * 100
