"""magmeas.MT class and functionality around it"""

import mammos_entity as me
import numpy as np
from scipy.signal import find_peaks

from magmeas.sweep import Sweep


class MT(Sweep):
    """
    Class for importing, storing and using of magnetometric data from M(T)-measurement
    aswell as derived properties.
    """

    def calc_Tc(
        self,
        segments=None,
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
        segments: TUPLE(NONE | INT), optional
            Segmentation points (start, end) at which the data used for the
            calculation will be cut off. This usually yields better results
            than using the whole dataset. Experience has shown that the return
            curve is often more useful, but this should be checked for each
            dataset.
            Default is None.
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
        T_key = self.get_entity_key("ThermodynamicTemperature")
        if isinstance(T_key, str):
            T = self[T_key]
        elif T_key is None:
            raise AttributeError(
                "No Entity with the ontology_label 'ThermodynamicTemperature' "
                + "could be found in self. Please provide one and try again."
            )
        else:
            raise AttributeError(
                "More than one Entity with the ontology_label "
                + "'ThermodynamicTemperature' could be found in self. "
                + "Please resolve this ambiguity, make sure only one is present "
                + "and try again."
            )
        if isinstance(self.get_entity_key("MagneticMoment"), str):
            m = self[self.get_entity_key("MagneticMoment")]
        elif isinstance(self.get_entity_key("Magnetization"), str):
            m = self[self.get_entity_key("Magnetization")]
        elif isinstance(self.get_entity_key("MagneticPolarisation"), str):
            m = self[self.get_entity_key("MagneticPolarisation")]
        elif isinstance(self.get_entity_key("MagneticMomentPerUnitMass"), str):
            m = self[self.get_entity_key("MagneticMomentPerUnitMass")]
        else:
            raise AttributeError(
                "The segments method needs exactly one entity with the ontology_label "
                + "'MagneticMoment', 'Magnetization' or 'MagneticPolarisation' "
                + "in self. Either none or too many were found. Please resolve "
                + "this and try again."
            )

        if segments is None:
            start = None
            end = None
        else:
            start, end = segments
        if start is not None:
            start = self.segments(edge_frac=edge_frac)[start]
        if end is not None:
            end = self.segments(edge_frac=edge_frac)[end]
        T = T[start:end]
        m = m[start:end]

        dMdT = self.dMdT(T.q, m.q, kernel_width, edge_frac)
        # norm dM/dT to values between -1 and 1
        ndMdT = (dMdT - np.nanmin(dMdT)) / (np.nanmax(dMdT) - np.nanmin(dMdT))
        # find peaks in normalised dM/dT
        p = find_peaks(-ndMdT, prominence=prominence)[0]
        # sort peaks
        Tc = np.sort(T.q[p])
        # pass all peaks of normalised dM/dT to Curie-Temperature
        # the user should decide themselves, which of them make sense and recalculate
        Tc = me.Entity("CurieTemperature", Tc, "K")
        self.Tc = Tc

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
        T_key = self.get_entity_key("ThermodynamicTemperature")
        if isinstance(T_key, str):
            T = self[T_key]
        elif T_key is None:
            raise AttributeError(
                "No Entity with the ontology_label 'ThermodynamicTemperature' "
                + "could be found in self. Please provide one and try again."
            )
        else:
            raise AttributeError(
                "More than one Entity with the ontology_label "
                + "'ThermodynamicTemperature' could be found in self. "
                + "Please resolve this ambiguity, make sure only one is present "
                + "and try again."
            )
        # find segmentation points
        # look for positive peak in T(t) (when measurement heats above RT)
        pos_peaks = find_peaks(T.q, prominence=prominence, **kwargs)[0]
        # look for negative peak in T(t) (when measurement cools below RT)
        neg_peaks = find_peaks(-T.q, prominence=prominence, **kwargs)[0]
        # unify all peaks in one segmentation point array
        s = np.sort(np.append(pos_peaks, neg_peaks))
        # discard segmentation points if they're very close to start
        s = s[len(s[s < (edge_frac * len(T.q))]) :]
        # discard segmentation points if they're very close to end
        if len(s[s > ((1 - edge_frac) * len(T.q))]) != 0:
            s = s[: -len(s[s > ((1 - edge_frac) * len(T.q))])]
        return s

    def dMdT(self, T, M, kernel_width, edge_frac):
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
