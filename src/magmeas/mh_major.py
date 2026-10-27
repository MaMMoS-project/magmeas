"""magmeas.MH_major class and functionality around it."""

import mammos_entity as me
import mammos_units as u
import matplotlib.pyplot as plt
import numpy as np
from mammos_analysis.hysteresis import extrinsic_properties
from scipy.signal import find_peaks
from scipy.stats import linregress

from magmeas._utils import _MH_unit_processing, _PropertyContainer
from magmeas.mh import MH


class MH_major(_PropertyContainer, MH):
    """
    Class for importing, storing and using of magnetometric data from major
    hysteresis loop M(H) measurements aswell as derived properties.
    """

    def __init__(self, datfile=None):
        super().__init__(datfile)
        if datfile is not None:
            self.calc_properties()

    def calc_properties(self, **kwargs):
        """
        Calculate extrinsic magnetic properties from the demagnetisation curve

        Parameters
        ----------
        kwargs: optional
            Keyword arguments used for the segmentation. See MH_major.segments
            for more details

        Returns
        -------
        None
        """
        self.properties = me.EntityCollection(
            "Sample properties derived from the measurement data."
        )
        s = self.segments(**kwargs)
        idx = np.argsort(self.H.q[s[0] : s[1]]) + s[0]
        prop = extrinsic_properties(self.H.q[idx], self.M.q[idx], 0)
        self.properties.remanence = me.Entity("Remanence", prop.Mr.q, "MA/m")
        self.properties.coercivity = me.Entity("CoercivityHc", prop.Hc.q, "MA/m")
        self.properties.BHmax = me.Entity("MaximumEnergyProduct", prop.BHmax.q, "kJ/m3")
        self.properties.kneefield = me.Entity(
            "KneeField", self._calc_kneefield(), "MA/m"
        )
        self.properties.squareness = self._calc_squareness()

    @property
    def remanence(self) -> me.Entity:
        """
        The remanence calculated from the measurement data.
        See mammos_analysis.hysteresis.extrinsic_properties for more details.
        """
        return self.properties.remanence

    @property
    def coercivity(self) -> me.Entity:
        """
        The coercivity calculated from the measurement data.
        See mammos_analysis.hysteresis.extrinsic_properties for more details.
        """
        return self.properties.coercivity

    @property
    def BHmax(self) -> me.Entity:
        """
        The maximum energy product calculated from the measurement data.
        See mammos_analysis.hysteresis.extrinsic_properties for more details.
        """
        return self.properties.BHmax

    @property
    def kneefield(self) -> me.Entity:
        """
        The kneefield calculated from the measurement data. This is the internal
        magnetic field at which a magnetisation of 90 % of the remanence is
        reached.
        """
        return self.properties.kneefield

    @property
    def squareness(self) -> float:
        """
        The hysteresis loop squareness factor calculated from the measurement
        data. It is calculated according to:
            .. math:: S = \\frac{H_k}{H_c}
        with Hk being the kneefield and Hc being the coercivity.
        """
        return self.properties.squareness

    @property
    def saturation(self) -> me.Entity:
        """
        The saturation magnetisation if it has been calculated. See the method
        `estimate_saturation` for more details.
        """
        return self.properties.saturation

    def _calc_kneefield(self):
        """
        Calculate kneefield of demagnetization curve. The knee field strength
        is the internal magnetic field at which the magnetization in the
        demagnetization curve is reduced to 90 % of the remanence.

        Parameters
        ----------
        None

        Returns
        -------
        kneefield: ENTITY
            Knee field strength as mammos_entity.Entity
        """
        s = self.segments()
        idx0 = np.argsort(self.M.q[s[0] : s[1]])
        Hk0 = np.interp(
            self.remanence.q * 0.9,
            self.M.q[s[0] : s[1]][idx0],
            self.H.q[s[0] : s[1]][idx0],
        )
        if len(s) >= 5:
            idx1 = np.argsort(self.M.q[s[2] : s[3]])
            Hk1 = np.interp(
                self.remanence.q * -0.9,
                self.M.q[s[2] : s[3]][idx1],
                self.H.q[s[2] : s[3]][idx1],
            )
            Hk = me.Entity("KneeField", max(np.abs(Hk) for Hk in [Hk0, Hk1]))
        else:
            Hk = me.Entity("KneeField", Hk0)
        return Hk

    def _calc_squareness(self):
        r"""
        Calculate squareness of demagnetization curve.

        .. math:: S = \frac{H_K}{H_C}

        Parameters
        ----------
        None

        Returns
        -------
        S: FLOAT
            Squareness (dimensionless)
        """
        return (self.kneefield.q / self.coercivity.q).value

    def estimate_saturation(self, threshold=None):
        """
        Calculate an estimation of the saturation magnetisation, return it and
        simultaneously assign it to the self. A possible high field
        susceptibility can optionally be corrected for.

        Parameters
        ----------
        threshold: NONE | FLOAT, optional
            If a float instead of None is given, the saturation value will be
            corrected for a high field susceptibility by substracting any
            inclination at magnetisation values above the maximum field times
            the threshold factor. The saturation magnetisation is then the
            y-intersection from extrapolating M over 1/H to 0.
            Default value is None.

        Returns
        -------
        Ms: ENTITY
            Saturation magnetisation.
        """
        if threshold is None:
            Ms = []
            for segment_num in range(0, len(self.segments()), 2):
                start_idx = self.segments()[segment_num]
                if start_idx == max(self.segments()):
                    end_idx = None
                else:
                    end_idx = self.segments()[segment_num + 1]
                Ms.append(max(np.abs(self.M.q[start_idx:end_idx])))

        if threshold is not None:
            Ms = []
            for segment_num in range(0, len(self.segments()), 2):
                start_idx = self.segments()[segment_num]
                if start_idx == max(self.segments()):
                    end_idx = None
                else:
                    end_idx = self.segments()[segment_num + 1]

                H = np.abs(self.H.q[start_idx:end_idx])
                M = np.abs(self.M.q[start_idx:end_idx])
                filt = np.max(H) * threshold < H
                linreg1 = linregress(H[filt], M[filt])
                linreg2 = linregress(1 / H[filt], M[filt] - H[filt] * linreg1.slope)
                Ms.append(u.Quantity(linreg2.intercept, "A/m"))

        Ms = me.Entity("SaturationMagnetization", max(Ms), "MA / m")
        self.properties.saturation = Ms
        return Ms

    def segments(self, edge_frac=0.05, prominence=1e6):
        r"""
        Find indices of segmentation points which can be used to seperate each
        measurement segment from each other. The segments are seperated by a
        changing sign of M (root) or a changing sign of dH/dt (peak).
        The segmentation points are chosen to be the index right after the root
        and right at the peak.
        Segmentation points with even indices (including 0) are peaks while
        segmentation points with uneven indices are roots.

        Parameters
        ----------
        edge_frac: FLOAT, optional
            Percentage of measurement to be treated as edge. Default is 0.05,
            which means that segmentation points closer than 5 % of the total
            measurement width to the edges will be discarded.
        prominence : FLOAT, optional
            The prominence which is used to find all peaks in H_ext(t). See the
            documentation of scipy.signal.find_peaks for more details.
            The default is 1e6.

        Returns
        -------
         s: ARRAY[INT]
            Array of segmentation points that can be used to segmentise the
            measurement.
        """
        t = self.t.q
        h = self.H_ext.q
        m = self.M.q
        # find all roots
        r = np.nonzero(np.diff(np.sign(m)))[0] + 2
        # find peaks, we know that they must be higher than 3e6 A/m and far apart
        p = find_peaks(np.abs(h), prominence=prominence)[0]
        # all segmentation points
        s = np.sort(np.append(r, p))
        # discard segmentation points if they're very close to start
        s = s[len(s[s < (edge_frac * len(t))]) :]
        # discard segmentation points if they're very close to end
        if len(s[s > ((1 - edge_frac) * len(t))]) != 0:
            s = s[: -len(s[s > ((1 - edge_frac) * len(t))])]
        return s

    def plot(
        self,
        filepath=None,
        demag=True,
        unit="T",
        segments=(None, None),
        fig_ax=None,
        **kwargs,
    ):
        """
        Plot major hysteresis loop, optionally with inset of demagnetization
        curve and save figure if a filepath is given.

        Parameters
        ----------
        filepath: STR | PATH, optional
            Filepath for saving the figure. Default is None, in that case no
            file is saved and the generated figure as well as all axes objects
            are returned.
        demag: BOOL, optional
            Boolean that determines if demagnetization curve is plotted as an
            inset next to hysteresis loop. Default is True.
        unit: STR | u.unit | TUPLE(STR | u.unit), optional
            Unit H and M are going to be plotted in. If string is given then
            both will have the same unit. Otherwise specify respective units
            as tuple (H_unit, M_unit). Default is 'T'.
        segments: TUPLE(NONE | INT), optional
            Segmentation points (start, end) at which the data to be plotted
            will be cut off. See the method 'segments' for more information.
            Note that valid input here are only the indices FOR the output from
            the segments method and NOT the indices of the actual measurement
            points. Use None to plot until the respective start or end. The
            demagnetisation curve will not be affected by this, it is always
            plotted in the same way between the same bounds.
            Default is (None, None).
        fig_ax: TUPLE(matplotlib.figure.Figure, matplotlib.axes.Axes), optional
            Tuple of figure and axes the data should be plotted to. If None is
            given they will be generated dynamically. Default is None.
        kwargs:
            Keyword arguments to change the appearance of plots.
            See documentation of matplotlib.pyplot.plot for more information.

        Returns
        -------
        None | (matplotlib.Figure, matplotlib.Axes)
            Return figure and axes object if filepath is not specified.
        """
        if fig_ax is None:
            fig, ax1 = plt.subplots()
        elif len(fig_ax) == 3:
            fig, ax1, ax2 = fig_ax
        else:
            fig, ax1 = fig_ax

        start, end = segments
        if start is not None:
            start = self.segments()[start]
        if end is not None:
            end = self.segments()[end]

        super().plot(
            filepath=None, unit=unit, bounds=(start, end), fig_ax=(fig, ax1), **kwargs
        )

        # plot inset of demagnetization curve
        if demag:
            H_unit, M_unit, _, _ = _MH_unit_processing(unit)
            # The demagnetisation curve will always be plotted in the same way
            start_idx, end_idx = self.segments()[:2]
            if "ax2" not in locals():
                ax2 = ax1.inset_axes([0.625, 0.15, 0.3, 0.5])
            ax2.plot(
                self.H.q.to(H_unit)[start_idx:end_idx],
                self.M.q.to(M_unit)[start_idx:end_idx],
                **kwargs,
            )

            # format inset
            Hmin = self.coercivity.q.to(H_unit).value * -1.1
            Jmax = self.remanence.q.to(M_unit).value * 1.1
            ax2.axis([Hmin, 0, 0, Jmax])
            ax2.yaxis.set_label_position("right")
            ax2.set_xlabel(ax1.get_xlabel())
            ax2.set_ylabel(ax1.get_ylabel())

        # save figure if filepath is given
        if filepath is not None:
            fig.savefig(filepath, dpi=300)
        if filepath is None:
            if demag:
                return fig, ax1, ax2
            else:
                return fig, ax1
