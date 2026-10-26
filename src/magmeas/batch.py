"""Handling of measurement batches."""

from importlib.metadata import version

import mammos_entity as me
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Colormap, ListedColormap

from magmeas._utils import _MH_unit_processing, _PropertyContainer
from magmeas.mh import MH
from magmeas.mh_major import MH_major
from magmeas.mt import MT


def plot_batch(
    data,
    filepath=None,
    labels=None,
    unit="T",
    segments=(None, None),
    cmap="inferno",
    color_range=(0, 0.9),
    demag=True,
    derivative=True,
    fig_ax=None,
    **kwargs,
):
    """
    Plot several magmeas objects together. Must be of the same type.

    Parameters
    ----------
    data: LIST[MH | MH_major | MT] | mammos_entity.EntityCollection
        List or EntityCollection of several objects which have to be of magmeas
        classes that have a plotting method. All contained objects need to be
        of the same class.
    filepath: STR | PATH, optional
        Filepath for saving the figure. If None is given, then instead the
        generated figure and axes objects will be returned. Default is None.
    labels: LIST, optional
        List of labels that are going to be used in the legend of the plot.
        Has to have the same length as data. If none is given then the names
        of the .DAT files each magmeas object was calculated from will be used.
    unit: STR | mu.unit | TUPLE(STR | mu.unit)
        Only supported for MH-derived measurements. Unit H and M are going to
        be plotted in. If string is given then both will have the same unit.
        Otherwise specify respective units as tuple (H_unit, M_unit).
        Default is 'T'.
    segments: TUPLE(NONE | INT), optional
        Segmentation points (start, end) at which the data to be plotted
        will be cut off. See the method 'segments' for more information.
        Note that valid input here are only the indices FOR the output from
        the segments method and NOT the indices of the actual measurement
        points. Use None to plot until the respective start or end. The
        demagnetisation curve will not be affected by this, it is always
        plotted in the same way between the same bounds.
        Default is (None, None).
    cmap: STR | matplotlib.colors.Colormap | matplotlib.colors.ListedColormap
        Colormap that will be used to generate the colours of all lines. Each
        colormap will be filled with np.linspace according to color_range.
        Please keep the visibility of ALL lines in mind! Colormaps such as
        'gray' or 'hot' will definitely lead to invisible/barely visible lines.
        Default is 'inferno'.
    color_range: TUPLE(FLOAT | INT), optional
        Color range to control from which range of the colormap gradient the
        colors will be picked. Full range would be (0, 1). Often it makes sense
        to restrict the range towards smaller maximum values as many colormaps
        have very bright colors at values getting closer to 1, which will lead
        to poorly visible lines. This can also be used to reverse the color
        gradient when switching maximum and minimum, as in (1, 0).
        Default is (0, 0.9).
    demag: BOOL, optional
        Determines, whether demagnetization curve is plotted as an inset next
        to hysteresis loop if MH_major objects are in data. Will not do
        anything otherwise. Default is True.
    derivative: BOOL, optional
        Determines, whether derivative dM/dT is plotted as another subplot
        below the M(T) plot if MT objects are in data. Will not do anything
        otherwise. Default is True.
    fig_ax: TUPLE(matplotlib.figure.Figure, matplotlib.axes.Axes), optional
        Tuple of figure and axes the data should be plotted to. If None is
        given they will be generated dynamically. Default is None.
    kwargs:
        Keyword arguments to change the appearance of lines. All arguments are
        applied to every line. The arguments 'color' and 'c' are not allowed as
        they would lead to all lines having the same colour, making them
        indistinguishable.
        See documentation of matplotlib.pyplot.plot for more information.

    Returns
    -------
    None | (matplotlib.Figure, matplotlib.Axes)
        Return figure and axes object if filepath is not specified.
    """
    if "color" in kwargs or "c" in kwargs:
        raise AttributeError(
            """Please do not pass 'color' or 'c' as a keyword argument. This
            would overwrite the colour of all lines, making them
            indistinguishable. Use the functionality of cmap instead."""
        )

    if isinstance(data, me.EntityCollection):
        data = [entry[1] for entry in data]

    ref_type = type(data[0])
    if len(set([type(vsm) for vsm in data])) != 1:
        wrong_type = np.where([type(vsm) is not ref_type for vsm in data])[0].tolist()
        raise TypeError(f"""
            The objects with the index {wrong_type} in data do not appear to be
            of the type {ref_type}, which was determined to be the reference
            type based on the first object in data. Please make sure that all
            objects in data are of the same type.""")

    if isinstance(cmap, str):
        cmap = colormaps.get_cmap(cmap)
    elif not isinstance(cmap, Colormap | ListedColormap):
        raise TypeError(
            f"cmap has type {type(cmap)}, which is not supported.\n"
            + "Please only pass a valid str or a Colormap from matplotlib to cmap."
        )
    colors = cmap(np.linspace(color_range[0], color_range[1], len(data)))

    if labels is None:
        labels = [vsm._path.stem for vsm in data]

    if ref_type == MH:
        fig, ax1 = data[0].plot(
            filepath=None,
            label=labels[0],
            color=colors[0],
            unit=unit,
            fig_ax=fig_ax,
            **kwargs,
        )
        for i in range(1, len(data)):
            data[i].plot(
                label=labels[i], color=colors[i], unit=unit, fig_ax=(fig, ax1), **kwargs
            )

    if ref_type == MH_major and not demag:
        fig, ax1 = data[0].plot(
            filepath=None,
            label=labels[0],
            color=colors[0],
            unit=unit,
            segments=segments,
            demag=demag,
            fig_ax=fig_ax,
            **kwargs,
        )
        for i in range(1, len(data)):
            data[i].plot(
                label=labels[i],
                color=colors[i],
                unit=unit,
                segments=segments,
                demag=demag,
                fig_ax=(fig, ax1),
                **kwargs,
            )

    if ref_type == MH_major and demag:
        fig, ax1, ax2 = data[0].plot(
            filepath=None,
            label=labels[0],
            color=colors[0],
            unit=unit,
            segments=segments,
            demag=demag,
            fig_ax=fig_ax,
            **kwargs,
        )
        for i in range(1, len(data)):
            data[i].plot(
                label=labels[i],
                color=colors[i],
                unit=unit,
                segments=segments,
                demag=demag,
                fig_ax=(fig, ax1, ax2),
                **kwargs,
            )
        H_unit, M_unit, _, _ = _MH_unit_processing(unit)
        Hmin = np.max([mh.coercivity.q.to(H_unit).value for mh in data]) * -1.1
        Jmax = np.max([mh.remanence.q.to(M_unit).value for mh in data]) * 1.1
        ax2.set_xlim([Hmin, 0])
        ax2.set_ylim([0, Jmax])

    if ref_type == MT and not derivative:
        fig, ax1 = data[0].plot(
            filepath=None,
            label=labels[0],
            color=colors[0],
            derivative=derivative,
            segments=segments,
            fig_ax=fig_ax,
            **kwargs,
        )
        for i in range(1, len(data)):
            data[i].plot(
                label=labels[i],
                color=colors[i],
                derivative=derivative,
                segments=segments,
                fig_ax=(fig, ax1),
                **kwargs,
            )

    if ref_type == MT and derivative:
        fig, ax1, ax2 = data[0].plot(
            filepath=None,
            label=labels[0],
            color=colors[0],
            derivative=derivative,
            segments=segments,
            fig_ax=fig_ax,
            **kwargs,
        )
        for i in range(1, len(data)):
            data[i].plot(
                label=labels[i],
                color=colors[i],
                derivative=derivative,
                segments=segments,
                fig_ax=(fig, ax1, ax2),
                **kwargs,
            )

    ax1.legend()

    if filepath is not None:
        fig.savefig(filepath, dpi=300)
    elif filepath is None and "ax2" in locals():
        return fig, ax1, ax2
    elif filepath is None and "ax2" not in locals():
        return fig, ax1


def to_batch(data, labels=None, prop_only=False):
    """Generate batch of magmeas objects as mammos_entity.EntityCollection.

    Parameters
    ----------
    data: LIST[MH_major | MT]
        List of magmeas objects that will be exported.
    labels: LIST[STR], optional
        List of strings to identify the magmeas objects more easily.
        If not given, the filenames of each magmeas object are used instead.
    prop_only: BOOL, optional
        Whether to only export the the properties and no measurement data. Will
        lead to an AttributeError if this is True but the objects in data only
        contain measurement data and no properties.
        Default is False.

    Returns
    -------
    None
    """
    if (
        not all([issubclass(type(vsm), _PropertyContainer) for vsm in data])
        and prop_only
    ):
        err_idx = np.where(
            [not issubclass(type(vsm), _PropertyContainer) for vsm in data]
        )[0].tolist()
        raise AttributeError(
            f"""The entries in data with the index {err_idx} do not seem to 
            contain any properties. Please only pass objects containing 
            properties to this function if you set 'prop_only=True'."""
        )

    description = (
        f"mammos-entity version = {version('mammos-entity')}\n"
        + f"magmeas version = {version('magmeas')}"
    )

    if labels is None:
        labels = [vsm._path.stem for vsm in data]

    if prop_only:
        data = [vsm.properties for vsm in data]

    batch = me.EntityCollection(
        description, **{label: vsm for label, vsm in zip(labels, data, strict=False)}
    )
    return batch


def to_yaml(data, filepath, labels=None, prop_only=False):
    """
    Save all properties derived from list of magmeas objects to YAML file.

    Parameters
    ----------
    data: LIST[MH_major | MT]
        List of magmeas objects that will be exported.
    filepath: STR | PATH
        Filepath to save the file to.
    labels: LIST[STR], optional
        List of strings to identify the magmeas objects more easily.
        If not given, the filenames of each magmeas objects are used instead.
    prop_only: BOOL, optional
        Whether to only export the the properties and no measurement data. Will
        lead to an AttributeError if this is True but the objects in data only
        contain measurement data and no properties.
        Default is False.

    Returns
    -------
    None
    """
    batch = to_batch(data=data, labels=labels, prop_only=prop_only)
    batch.to_yaml(filepath)


def to_hdf5(data, filepath, labels=None, prop_only=False):
    """
    Save all properties derived from list of magmeas objects to HDF5 file.

    Parameters
    ----------
    data: LIST[MH_major | MT]
        List of magmeas objects that will be exported.
    filepath: STR | PATH
        Filepath to save the file to.
    labels: LIST[STR], optional
        List of strings to identify the magmeas objects more easily.
        If not given, the filenames of each magmeas objects are used instead.
    prop_only: BOOL, optional
        Whether to only export the the properties and no measurement data. Will
        lead to an AttributeError if this is True but the objects in data only
        contain measurement data and no properties.
        Default is False.

    Returns
    -------
    None
    """
    batch = to_batch(data=data, labels=labels, prop_only=prop_only)
    batch.to_hdf5(filepath)
