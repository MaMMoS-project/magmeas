"""magmeas.MH_recoil class and functionality around it"""

import mammos_units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress

from magmeas.mh import MH


class MH_recoil(MH):
    """
    Class for importing, storing and using of magnetometric data from recoil loop M(H)
    measurements.
    Recoil loop measurements are assumed to be measurements where
    the sample is first saturated at some positive field. After this, a number
    of recoil loops are performed, where the magnetic field is reduced to some
    negative value (which often includes fields that do not fully demagnetise
    the sample) and is then increased to an external field of 0 or to a
    positive external field which increases the internal field to 0.
    """

    def segments(self, prominence=5e3):
        """
        Get indices of segmentation points. Segmentation point 0 occurs
        at H = H_max. All uneven segmentation points are the start of a recoil
        loop (only section with positive change in H) while all subsequent even
        segmentation points are the respective ends of each recoil loop.

        Parameters
        ----------
        prominence : FLOAT, optional
            The prominence which is used to find all peaks in H_ext(t). See the
            documentation of scipy.signal.find_peaks for more details.
            The default is 5e3.

        Returns
        -------
        segments: ARRAY
            Segmentation points used to extract each recoil loop.
        """
        npeaks, _ = find_peaks(-self.H_ext.q, prominence=prominence)
        # npeaks = npeaks[self.H.q[npeaks] < 0] + 1
        npeaks = npeaks + 1
        ppeaks, _ = find_peaks(self.H_ext.q, prominence=prominence)
        # ppeaks = ppeaks[self.H.q[ppeaks] < 0]
        ppeaks = np.append(ppeaks, len(self.H.q) - 1)
        apeaks = np.sort(np.append(npeaks, ppeaks))
        return apeaks

    def recoil_susceptibility(self, prominence=5e3, take_mean=True):
        """
        Calculate recoil susceptibility from recoil loop measurement as slope
        of a linear regression of each recoil loop.

        Parameters
        ----------
        prominence: FLOAT, optional
            Prominence used to find segmentation points, see MH_recoil.segments
            The default is 5e3.
        take_mean: BOOL, optional
            Wether to take the mean of all recoil susceptibilities instead of
            returning an individual one for each loop. Default is True.

        Returns
        -------
        recoil_susceptibility: FLOAT | ARRAY
            Recoil susceptibility calculated from recoil loop measurement.
        """
        segments = self.segments(prominence=prominence)
        starts = segments[1::2]
        ends = segments[2::2]
        num_loops = min([len(starts), len(ends)])

        recoil_suscep = np.array(
            [
                linregress(
                    self.H.q[starts[i] : ends[i]], self.M.q[starts[i] : ends[i]]
                ).slope
                for i in range(num_loops)
            ]
        )

        if take_mean:
            recoil_suscep = np.mean(recoil_suscep)

        return recoil_suscep

    def external_comp_field(self, demag_fields=None, prominence=5e3):
        r"""
        Return external compensation fields, that will lead to an endpoint
        close to an internal magnetic field of zero for each recoil loop. This
        only makes sense to be calculated from recoil loop measurements, where
        the endpoint for each recoil loop was an external magnetic field of 0,
        which will lead to the respective internal magnetic field being
        non-zero. The external compensation fields can then be used in a new
        measurement which will then enable a recoil loop measurement corrected
        for demagnetisation.\n
        External compensation field is calculated from the internal recoil
        fields (at the end of each recoil loop) and the recoil susceptibility
        according to:

            .. math:: H^e_{comp} = \frac{- H^i_{recoil}}{1 - D * \chi_{recoil}}

        Parameters
        ----------
        demag_fields: NONE | QUANTITY | ARRAY, optional
            If None, then the external compensation fields will be calculated
            for each recoil loop of the given MH_recoil object. Otherwise an
            array of external demagnetising fields can be given, which will
            be the start of each recoil loop for which the external compensation
            field is calculated. Must be given as a Quantity, otherwise its
            unit is assumed to be A/m. Default is None.
        prominence: FLOAT, optional
            Prominence used to find segmentation points, see MH_recoil.segments
            The default is 5e3.

        Returns
        -------
        external_compensation_field: ARRAY
            External compensation fields, that would lead to an internal field
            close to zero at the end of each recoil loop.
        """
        # get demagnetising factor
        D = self.D.value
        # get internal recoil fields as end points of each recoil loop
        i_r_f = self.H.q[self.segments(prominence=prominence)[2::2]]
        # get recoil susceptibilities individually for each recoil loop
        r_chi = self.recoil_susceptibility(
            prominence=prominence, take_mean=demag_fields is not None
        )

        if demag_fields is not None:
            e_d_f = self.H_ext.q[self.segments(prominence=prominence)[1::2]]
            fit = np.polynomial.Polynomial.fit(e_d_f.value, i_r_f.value, 2)
            if isinstance(demag_fields, u.Quantity):
                demag_fields = demag_fields.to("A/m").value
            i_r_f = fit(demag_fields)

        # calculate external compensation fields
        e_c_f = -i_r_f / (1 - D * r_chi)

        return e_c_f

    def plot(
        self,
        filepath=None,
        demag=True,
        unit="T",
        segments=(0, None),
        fig_ax=None,
        **kwargs,
    ):
        """
        Plot recoil loop measurement and save figure if a filepath is given.

        Parameters
        ----------
        filepath: STR | PATH, optional
            Filepath for saving the figure. Default is None, in that case no
            file is saved and the generated figure as well as all axes objects
            are returned.
        demag: BOOL, optional
            Plot the recoil loop measurement only in the second quadrant.
            Default is True.
        unit: STR | u.unit | TUPLE(STR | u.unit), optional
            Unit H and M are going to be plotted in. If string is given then
            both will have the same unit. Otherwise specify respective units
            as tuple (H_unit, M_unit). Default is 'T'.
        segments: TUPLE(NONE | INT), optional
            Segmentation points (start, end) at which the data to be plotted
            will be cut off. See the method 'segments' for more information.
            Note that valid input here are only the indices FOR the output from
            the segments method and NOT the indices of the actual measurement
            points. Use None to plot until the respective start or end.
            Default is (0, None).
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
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax

        start, end = segments
        if start is not None:
            start = self.segments()[start]
        if end is not None:
            end = self.segments()[end]

        super().plot(
            filepath=None, unit=unit, bounds=(start, end), fig_ax=(fig, ax), **kwargs
        )
