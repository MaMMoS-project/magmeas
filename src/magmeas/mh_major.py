"""magmeas.MH_major class and functionality around it."""

import mammos_entity as me
import mammos_units as u
import numpy as np
from mammos_analysis.hysteresis import extrinsic_properties
from scipy.signal import find_peaks
from scipy.stats import linregress

from magmeas.sweep import Sweep


class MH_major(Sweep):
    """
    Class for importing, storing and using of magnetometric data from major
    hysteresis loop M(H) measurements aswell as derived properties.
    """

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
        magnetisation_key = self.get_entity_key("Magnetization")
        int_field_key = self.get_entity_key("InternalMagneticField")
        if magnetisation_key is None or int_field_key is None:
            raise AttributeError(
                "Either the Magnetization or the InternalMagneticField could "
                + "not be found in self. This is currently not yet supported."
            )
        s = self.segments(**kwargs)
        idx = np.argsort(self.H.q[s[0] : s[1]]) + s[0]
        prop = extrinsic_properties(self.H.q[idx], self.M.q[idx], 0)
        self.Mr = me.Entity("Remanence", prop.Mr.q, "MA/m")
        self.Hc = me.Entity("CoercivityHc", prop.Hc.q, "MA/m")
        self.BHmax = me.Entity("MaximumEnergyProduct", prop.BHmax.q, "kJ/m3")
        self.Hk = me.Entity("KneeField", self.calc_Hk(), "MA/m")
        self.SF = me.Entity("LoopSquarenessFactorInternal", self.calc_SF())

    def calc_Hk(self, **kwargs):
        """
        Calculate kneefield of demagnetization curve. The knee field strength
        is the internal magnetic field at which the magnetization in the
        demagnetization curve is reduced to 90 % of the remanence.

        Parameters
        ----------
        kwargs: optional
            Keyword arguments used for the segmentation. See MH_major.segments
            for more details

        Returns
        -------
        Hk: ENTITY
            Knee field strength as mammos_entity.Entity
        """
        s = self.segments()
        idx0 = np.argsort(self.M.q[s[0] : s[1]])
        Hk0 = np.interp(
            self.Mr.q * 0.9,
            self.M.q[s[0] : s[1]][idx0],
            self.H.q[s[0] : s[1]][idx0],
        )
        if len(s) >= 5:
            idx1 = np.argsort(self.M.q[s[2] : s[3]])
            Hk1 = np.interp(
                self.Mr.q * -0.9,
                self.M.q[s[2] : s[3]][idx1],
                self.H.q[s[2] : s[3]][idx1],
            )
            Hk = me.Entity("KneeField", max(np.abs(Hk) for Hk in [Hk0, Hk1]))
        else:
            Hk = me.Entity("KneeField", Hk0)
        return Hk

    def calc_SF(self):
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
        return (self.Hk.q / self.Hc.q).value

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

    def segments(self, edge_frac=0.05, prominence=1e6, **kwargs):
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
        prominence: FLOAT, optional
            The prominence which is used to find all peaks in H_ext(t). See the
            documentation of scipy.signal.find_peaks for more details.
            The default is 1e6.
        kwargs: optional
            Additional keyword arguments that can be used for peak finding.
            See scipy.signal.find_peaks for more details

        Returns
        -------
        s: ARRAY[INT]
            Array of segmentation points that can be used to segmentise the
            measurement.
        """
        if isinstance(self.get_entity_key("ExternalMagneticField"), str):
            h = self[self.get_entity_key("ExternalMagneticField")]
        elif isinstance(self.get_entity_key("InternalMagneticField"), str):
            h = self[self.get_entity_key("InternalMagneticField")]
        else:
            raise AttributeError(
                "The segments method needs exactly one entity with the ontology_label "
                + "'ExternalMagneticField' or 'InternalMagneticField' in self. "
                + "Either none or too many were found. Please resolve this and "
                + "try again."
            )
        if isinstance(self.get_entity_key("MagneticMoment"), str):
            m = self[self.get_entity_key("MagneticMoment")]
        elif isinstance(self.get_entity_key("Magnetization"), str):
            m = self[self.get_entity_key("Magnetization")]
        elif isinstance(self.get_entity_key("MagneticPolarisation"), str):
            m = self[self.get_entity_key("MagneticPolarisation")]
        else:
            raise AttributeError(
                "The segments method needs exactly one entity with the ontology_label "
                + "'MagneticMoment', 'Magnetization' or 'MagneticPolarisation' "
                + "in self. Either none or too many were found. Please resolve "
                + "this and try again."
            )
        # find all roots
        r = np.nonzero(np.diff(np.sign(m.q)))[0] + 2
        # find peaks, we know that they must be higher than 3e6 A/m and far apart
        p = find_peaks(np.abs(h.q), prominence=prominence, **kwargs)[0]
        # all segmentation points
        s = np.sort(np.append(r, p))
        # discard segmentation points if they're very close to start
        s = s[len(s[s < (edge_frac * len(h.q))]) :]
        # discard segmentation points if they're very close to end
        if len(s[s > ((1 - edge_frac) * len(h.q))]) != 0:
            s = s[: -len(s[s > ((1 - edge_frac) * len(h.q))])]
        return s
