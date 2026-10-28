"""magmeas.MT class and functionality around it"""

import mammos_entity as me
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from magmeas.sweep import Sweep


class MT(Sweep):
    """
    Class for importing, storing and using of magnetometric data from M(T)-measurement
    aswell as derived properties.
    """

    def __init__(self, filepath=None):
        # import data
        super().__init__(filepath)

        # Initialise property collection
        self.properties = me.EntityCollection(
            "Sample properties derived from the measurement data."
        )
        if filepath is not None:
            # calculate properties
            self.calc_Tc()

    @property
    def Tc(self) -> me.Entity:
        """The Curie-temperature(s) calculated from the measurement data."""
        return self.properties.Tc

    def calc_Tc(
        self,
        save=True,
        segments=(0, None),
        kernel_width=40,
        edge_frac=0.05,
        prominence=0.2,
        **kwargs,
    ):
        """
        Calculate Curie-temperature from M(T) measurement. The results should
        always be checked for feasability and ideally compared with a plot of
        M(T) and dM/dT(T) to confirm. Curie-temperatures are assumed to occur
        at turning points of the M(T) curve.

        Parameters
        ----------
        save: BOOL, optional
            Whether to save the calculated Curie-Temperature(s) in the MT object
            or just return them.
            Default is True.
        segments: TUPLE(NONE | INT), optional
            Segmentation points (start, end) at which the data used for the
            calculation will be cut off. This is usually yields better results
            than using the whole dataset. Experience has shown that the cooling
            curve is often more useful, but this should be checked for each
            dataset.
            Default is (0, None).
        kernel_width: INT, optional
            Kernel width to be used for smoothing of derivative.
            Default is 40.
        edge_frac: FLOAT, optional
            Fraction of measurement range to be masked at the edges, since those
            regions are usually very noisy. This is combined with the segments
            argument with the edges being cut off after the segments have been
            selected.
            Default is 0.05.
        prominence: FLOAT
            Prominence to be used in the peak finding step.
            See documentation of scipy.signal.find_peaks for more information.
            Default is 0.2.
        kwargs:
            Keyword arguments to be used for improved peak finding.
            See documentation of scipy.signal.find_peaks for more information.

        Returns
        -------
        Tc: ENTITY
            Curie-Temperature as mammos_entity.Entity
        """
        start, end = segments
        if start is not None:
            start = self.segments(edge_frac=edge_frac)[start]
        if end is not None:
            end = self.segments(edge_frac=edge_frac)[end]

        T = self.T.q[start:end]
        M = self.M.q[start:end]
        dMdT = _dMdT(T, M, kernel_width, edge_frac)
        # norm dM/dT to values between -1 and 1
        ndMdT = (dMdT - np.nanmin(dMdT)) / (np.nanmax(dMdT) - np.nanmin(dMdT))
        # find peaks in normalised dM/dT
        p = find_peaks(-ndMdT, prominence=prominence)[0]
        # sort peaks
        Tc = np.sort(T[p])
        # pass all peaks of normalised dM/dT to Curie-Temperature
        # the user should decide themselves, which of them make sense and recalculate
        Tc = me.Entity("CurieTemperature", Tc)
        if save:
            self.properties.Tc = Tc
        return Tc

    def segments(self, edge_frac=0.05, prominence=5, **kwargs):
        r"""
        Find indices of segmentation points which can be used to seperate each
        measurement segment from each other. The segments are seperated by a
        changing sign of dT/dt.

        Parameters
        ----------
        edge_frac: FLOAT, optional
            Fraction of measurement to be treated as edge. Default is 0.05,
            which means that segmentation points closer than 5 % of the total
            measurement width to the edges will be discarded.
            Default is 0.05.
        prominence: INT | FLOAT, optional
            Prominence of peaks in T(t) or -T(t) which will be used to define
            segemntation points.
            Default is 5.
        kwargs: optional
            Optional keyword arguments that will be passed to the peak finding
            function.
            See scipy.signal.find_peaks for more information.

        Returns
        -------
         s: ARRAY[INT]
            Array of segmentation points that can be used to segmentise the
            measurement.
        """
        t = self.t.q
        T = self.T.q
        # find segmentation points
        # look for positive peak in T(t) (when measurement heats above RT)
        pos_peaks = find_peaks(T, prominence=prominence, **kwargs)[0]
        # look for negative peak in T(t) (when measurement cools below RT)
        neg_peaks = find_peaks(-T, prominence=prominence, **kwargs)[0]
        # unify all peaks in one segmentation point array
        s = np.sort(np.append(pos_peaks, neg_peaks))
        # discard segmentation points if they're very close to start
        s = s[len(s[s < (edge_frac * len(t))]) :]
        # discard segmentation points if they're very close to end
        if len(s[s > ((1 - edge_frac) * len(t))]) != 0:
            s = s[: -len(s[s > ((1 - edge_frac) * len(t))])]
        return s

    def plot(
        self,
        filepath=None,
        derivative=True,
        segments=(None, None),
        fig_ax=None,
        kernel_width=40,
        edge_frac=0.05,
        **kwargs,
    ):
        """
        Plot cooling curve of M(T) measurement. Save to file if path is given.

        Parameters
        ----------
        filepath : STR | PATH, optional
            Filepath for saving the figure. Default is None, in that case no
            file is saved.
        derivative: BOOL, optional
            Whether to plot the first derivative dM/dT on a second axes.
            Default is True.
        segments: TUPLE(NONE | INT), optional
            Segmentation points (start, end) at which the data to be plotted
            will be cut off. See the method 'segments' for more information.
            Note that valid input here are only the indices FOR the output from
            the segments method and NOT the indices of the actual measurement
            points. Use None to plot until the respective start or end.
            Default is (None, None).
        fig_ax: TUPLE(matplotlib.figure.Figure, matplotlib.axes.Axes), optional
            Tuple of figure and axes the data should be plotted to. If None is
            given they will be generated dynamically.
            Default is None.
        kernel_width: INT, optional
            Kernel width to be used for smoothing of derivative.
            Default is 40.
        edge_frac: FLOAT, optional
            Fraction of measurement range to be masked at the edges, since those
            regions are usually very noisy.
            Default is 0.05.
        kwargs:
            Keyword arguments to change the appearance of plots.
            See documentation of matplotlib.pyplot.plot for more information.

        Returns
        -------
        None | (matplotlib.Figure, matplotlib.Axes)
            Return figure and axes object if filepath is not specified.
        """
        if derivative and fig_ax is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        elif derivative and fig_ax is not None:
            fig, ax1, ax2 = fig_ax
        elif not derivative and fig_ax is None:
            fig, ax1 = plt.subplots(1, 1)
        elif not derivative and fig_ax is not None:
            fig, ax1 = fig_ax

        start, end = segments
        if start is not None:
            start = self.segments(edge_frac=edge_frac)[start]
        if end is not None:
            end = self.segments(edge_frac=edge_frac)[end]

        T = self.T.q[start:end]
        M = self.M.q[start:end]

        ax1.plot(T, M, **kwargs)
        ax1.set_xlabel(f"Temperature in {self.T.unit}")
        ax1.set_ylabel(f"Magnetization in {self.M.unit}")

        if derivative:
            dMdT = _dMdT(T, M, kernel_width, edge_frac)
            ax2.plot(T, dMdT, **kwargs)
            ax2.set_ylabel("$\\frac{dM}{dT}$")
            ax2.set_xlabel(f"Temperature in {self.T.unit}")

        if filepath is not None:
            fig.savefig(filepath, dpi=300)
        if filepath is None:
            if derivative:
                return fig, ax1, ax2
            else:
                return fig, ax1


def _dMdT(T, M, kernel_width, edge_frac):
    """
    Return first derivative of magnetisation over temperature. Includes
    smoothing and masking functionality.

    Parameters
    ----------
    T: ARRAY | QUANTITY
        Thermodynamic temperature.
    M: ARRAY | QUANTITY
        Magnetisation or magnetic polarisation.
    kernel_width: INT
        Width of kernel to be used for smoothing. Should be even value.
    edge_frac: FLOAT
        Fraction of the array to be masked off at the edges.

    Returns
    -------
    dMdT: QUANTITY
        First derivative of magnetisation over temperature.
    """
    if kernel_width % 2 != 0:
        raise ValueError(
            """Kernel width was detected to be uneven! Please only use even
            integer kernel widths."""
        )
    if not isinstance(kernel_width, int):
        raise TypeError(
            f"""Kernel width is expected to be of type int, but type
            {type(kernel_width)} was used."""
        )
    if edge_frac is not None and not (0 <= edge_frac <= 1):
        raise ValueError(
            f"""Please only use values of edge_frac between 0 and 1.
            edge_frac was {edge_frac}."""
        )
    kernel = np.ones(kernel_width) / kernel_width
    sT = np.convolve(T, kernel, mode="same")
    sM = np.convolve(M, kernel, mode="same")

    dT = np.gradient(sT)
    dM = np.gradient(sM)
    with np.errstate(divide="ignore", invalid="ignore"):
        dMdT = dM / dT
    dMdT = np.where(np.isfinite(dMdT), dMdT, np.nan)

    if edge_frac:
        edge_len = (np.max(T) - np.min(T)) * edge_frac
        dMdT[np.min(T) + edge_len >= T] = np.nan
        dMdT[np.max(T) - edge_len <= T] = np.nan
    return dMdT
