"""magmeas.FORC class and functionality around it"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from magmeas.mh import MH


class FORC(MH):
    """Class for importing, storing and using VSM data of FORC measurements."""

    def __init__(self, datfile, read_method="auto"):
        super().__init__(datfile, read_method="auto")

    def segments(self, prominence=5e3):
        """
        Get indices of segmentation points. Segmentation point 0 occurs
        at H = H_max. All uneven segmentation points are the start of a FORC
        (only section with positive change in H) while all subsequent even
        segmentation points are the respective ends of each FORC.

        Parameters
        ----------
        prominence: FLOAT, optional
            The prominence which is used to find all peaks in H_ext(t). See the
            documentation of scipy.signal.find_peaks for more details.
            The default is 5e3.

        Returns
        -------
        segments: ARRAY
            Segmentation points used to extract each recoil loop.
        """
        ppeaks, _ = find_peaks(self.H_ext.q, prominence)
        ppeaks = np.append(ppeaks, len(self.H_ext.q) - 1)
        npeaks, _ = find_peaks(-self.H_ext.q + np.max(self.H_ext.q), prominence)
        npeaks = npeaks + 1
        apeaks = np.sort(np.append(ppeaks, npeaks))
        return apeaks

    def forc_grid(self, demag_correction=False, prominence=5e3):
        """
        Extract grid of magnetic fields, reversal fields and magnetisations
        from the FORC-measurement.

        Parameters
        ----------
        demag_correction: BOOL, optional
            Whether to correct the magnetic field for Demagnetisation.
            The default is True.
        prominence: FLOAT, optional
            Prominence used to find segmentation points, see FORC.segments
            The default is 5e3.

        Returns
        -------
        h_grid: QUANTITY
            Magnetic field at each grid point of the FORC-measurement.
            If demag_correction was True, this is the internal magnetic field.
            Otherwise it's the external magnetic field.
        h_r_grid: QUANTITY
            Magnetic reversal field of each FORC. Has the same shape as h_grid.
            If demag_correction was True, this is the internal magnetic field.
            Otherwise it's the external magnetic field.
        m_grid: QUANTITY
            Magnetisation at each grid point of the FORC-measurement.
        """
        if demag_correction:
            h = self.H.q
        else:
            h = self.H_ext.q
        m = self.M.q

        # get starting and end points for each FORC
        starts = self.segments(prominence=prominence)[1::2]
        ends = self.segments(prominence=prominence)[2::2]
        # find highest number of points found in any FORC, which is grid width
        grid_width = np.max([ends[i] - starts[i] for i in range(len(starts))])

        # make sure all FORCs have start and end point
        # total number of FORCs will be grid height
        if starts.shape == ends.shape:
            grid_height = starts.shape[0]
        else:
            raise Exception(
                "Grid height cannot be calculated. Number of"
                + " positive and negative peaks is not the same"
            )

        # initialise grids so they can be referred to in the loop
        h_grid = np.array([]) * h.unit
        h_r_grid = np.array([]) * h.unit
        m_grid = np.array([]) * m.unit

        # loop over each FORC
        for i in range(len(starts)):
            # prepare pad of nan, so that each FORC will have same (grid) width
            pad = np.array((grid_width - (ends[i] - starts[i])) * [np.nan])

            # get all H of current FORC and add pad (pad can be empty)
            current_h = np.append(pad * h.unit, h[starts[i] : ends[i]])
            # add H of current FORC to grid of all FORCs
            h_grid = np.append(h_grid, current_h)

            # get reversal field H_r of current FORC
            current_h_r = h[starts[i]]
            # H_r is the same for all points (H) of one FORC
            current_h_r = current_h_r.repeat(ends[i] - starts[i])
            # add pad to reach grid width
            current_h_r = np.append(pad * h.unit, current_h_r)
            # add H_r of current FORC to grid of all FORCs
            h_r_grid = np.append(h_r_grid, current_h_r)

            # get all M of current FORC and add pad (pad can be empty)
            current_m = np.append(pad * m.unit, m[starts[i] : ends[i]])
            # add M of current FORC to grid of all FORCs
            m_grid = np.append(m_grid, current_m)

        # reshape all flat datasets prepared in loop to actual grids
        h_grid = h_grid.reshape((grid_height, grid_width))
        h_r_grid = h_r_grid.reshape((grid_height, grid_width))
        m_grid = m_grid.reshape((grid_height, grid_width))

        return h_grid, h_r_grid, m_grid

    def forc_dist(self, SF, demag_correction=True, prominence=5e3):
        """
        Calculate the transformed coordinates H_c and H_i as well as the
        FORC distribution. Takes a grid of (2*SF+1)**2 measurement points from
        the grid of M(H_r, H) points and fits a quadratic function. The
        FORC distribution is then the mixed partial derivative of this fitted
        polynomial.

        Parameters
        ----------
        SF: INT
            Smoothing factor to be used in the calculation of the FORC
            distribution. The number of points contributing to a point in the
            FORC distribution is a square grid with a side length of (2*SF+1)
            measurement points.
        demag_correction: BOOL, optional
            Whether to correct the magnetic field for Demagnetisation.
            The default is True.
        prominence: FLOAT, optional
            Prominence used to find segmentation points, see FORC.segments
            The default is 5e3.

        Returns
        -------
        H_c: QUANTITY
            Local coercive field H_c = (H_r - H) / 2 defined for H > 0
        H_i: QUANTITY
            Local interaction field H_i = (H_r + H) / 2
        rho: ARRAY
            FORC distribution at each point of the FORC measurement grid.
        """

        # define quadratic function that will be fitted to measurement points
        def quadratic_fun(h_hr, a1, a2, a3, a4, a5, a6):
            h, hr = h_hr
            return (
                a1 + a2 * hr + a3 * hr**2 + a4 * h + a5 * h**2 + a6 * hr * h
            ).ravel()

        # get the output from FORC.forc_grid as input for this method
        h, hr, m = self.forc_grid(
            demag_correction=demag_correction, prominence=prominence
        )

        # initialise empty forc_distribution to be used in loops
        rho = np.zeros(m.shape)
        rho[:, :] = np.nan

        # loop over every FORC
        for i in range(SF, m.shape[0] - SF):
            # loop over every point in each FORC
            for j in range(SF, m.shape[1] - SF + 1):
                # extract all points that contribute to FORC-distribution
                h_grid = h[i - SF : i + SF + 1, j - SF : j + SF + 1]
                hr_grid = hr[i - SF : i + SF + 1, j - SF : j + SF + 1]
                m_grid = m[i - SF : i + SF + 1, j - SF : j + SF + 1]

                # don't calculate a FORC-distribution if any of the points in
                # the selected grid are undefined
                if np.isnan([h_grid, hr_grid, m_grid]).any():
                    continue

                # calculate FORC-distribution at selected point
                else:
                    # a6 is the last of the fitted parameters
                    a6 = curve_fit(quadratic_fun, (h_grid, hr_grid), m_grid.ravel())[0][
                        -1
                    ]
                    rho[i, j] = a6 * -0.5

        # calculate coordinate transformation from (H, H_r) to (H_c, H_i)
        H_c = np.abs((hr - h) / 2)
        H_i = (hr + h) / 2

        return H_c, H_i, rho
