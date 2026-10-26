"""magmeas.MH class and functionality around it."""

import matplotlib.pyplot as plt

from magmeas._utils import _MH_unit_processing
from magmeas.sweep import Sweep


class MH(Sweep):
    """
    Class for importing, storing and using of magnetometric data from M(H)
    measurements.
    """

    def plot(self, filepath=None, unit="T", bounds=(None, None), fig_ax=None, **kwargs):
        """
        Plot M(H) measurement and save figure if a filepath is given.

        Parameters
        ----------
        filepath: STR | PATH, optional
            Filepath for saving the figure. Default is None, in that case no
            file is saved and the generated figure as well as all axes objects
            are returned.
        unit: STR | mu.unit | TUPLE(STR | mu.unit), optional
            Unit H and M are going to be plotted in. If string is given then
            both will have the same unit. Otherwise specify respective units
            as tuple (H_unit, M_unit). Default is 'T'.
        bounds: TUPLE(NONE | INT), optional
            Indices (start, end) at which the data to be plotted will be cut
            off. Use None to plot until the respective start or end.
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
        H_unit, M_unit, H_label, M_label = _MH_unit_processing(unit)

        start, end = bounds

        if fig_ax is None:
            fig, ax1 = plt.subplots()
        else:
            fig, ax1 = fig_ax

        ax1.plot(
            self.H.q.to(H_unit)[start:end], self.M.q.to(M_unit)[start:end], **kwargs
        )

        # format plot
        ax1.set_xlabel(f"{H_label} in {H_unit}")
        ax1.set_ylabel(f"{M_label} in {M_unit}")

        # save figure if filepath is given
        if filepath is not None:
            fig.savefig(filepath, dpi=300)
        if filepath is None:
            return fig, ax1
