"""Testing routine for plotting of magmeas.VSM-derived objects."""

from pathlib import Path

import mammos_units as mu
import matplotlib
import pytest

import magmeas

matplotlib.use("Agg")
cwd = Path(__file__).parent.resolve()


def test_MH_io():
    """Test export of MH-object to PNG."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH(file_path)
    output_path = cwd.joinpath("VSM_MH.png")
    try:
        mh.plot(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()
    matplotlib.pyplot.close()


def test_MH_return():
    """Test return of matplotlib Figure and Axes objects from MH.plot."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH(file_path)
    try:
        fig, ax = mh.plot()
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_MH_figax():
    """Test MH.plot with previously generated input Figure and Axes objects."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH(file_path)
    input_fig, input_ax = matplotlib.pyplot.subplots()
    try:
        fig, ax = mh.plot(fig_ax=(input_fig, input_ax))
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_MH_unit_str():
    """Test MH.plot with argument unit as string."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH(file_path)
    try:
        fig, ax = mh.plot(unit="kA/m")
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_MH_unit_mu():
    """Test MH.plot with argument unit as mammos_units.CompositeUnit."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH(file_path)
    try:
        fig, ax = mh.plot(unit=mu.MA / mu.m)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_MH_unit_tuple2():
    """Test MH.plot with argument unit as tuple of string and
    mammos_units.CompositeUnit.
    """
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH(file_path)
    try:
        fig, ax = mh.plot(unit=("kA/m", mu.mT))
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_MH_unit_tuple1():
    """Test MH.plot with argument unit as tuple of length 1."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH(file_path)
    try:
        fig, ax = mh.plot(unit=("kA/m"))
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_MH_kwargs():
    """Test MH.plot with keyword arguments passed to matplotlibs plot."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH(file_path)
    try:
        fig, ax = mh.plot(color="tab:green", linestyle=":", marker="o")
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_MH_ValueError_compatible_unit():
    """Test occurence of ValueError when an incompatible unit is passed as the
    unit argument.
    """
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH(file_path)
    with pytest.raises(ValueError):
        mh.plot(unit="kg")
    matplotlib.pyplot.close()


def test_MH_ValueError_unit_tuple_len():
    """Test occurence of ValueError when the length of unit exceeds 2 if it is
    a tuple or similar.
    """
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH(file_path)
    with pytest.raises(ValueError):
        mh.plot(unit=("T", "T", "T"))
    matplotlib.pyplot.close()


def test_MH_major_io():
    """Test export of MH_major-object to PNG."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)
    output_path = cwd.joinpath("VSM_MH.png")
    try:
        mh.plot(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()
    matplotlib.pyplot.close()


def test_MH_major_return2():
    """Test return of matplotlib Figure and Axes objects from MH_major.plot if
    demag==False.
    """
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)
    try:
        fig, ax = mh.plot(demag=False)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_MH_major_return3():
    """Test return of matplotlib Figure and Axes objects from MH_major.plot if
    demag==True.
    """
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)
    try:
        fig, ax, ax1 = mh.plot(demag=True)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(ax1, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_MH_major_figax():
    """Test MH_major.plot with previously generated input Figure and Axes
    objects.
    """
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)
    input_fig, (input_ax1, input_ax2) = matplotlib.pyplot.subplots(1, 2)
    try:
        fig, ax, ax1 = mh.plot(fig_ax=(input_fig, input_ax1, input_ax2))
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(ax1, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_MH_major_unit():
    """Test handling of argument unit in MH_major.plot without Error."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)
    try:
        fig, ax, ax1 = mh.plot(unit=(mu.kA / mu.m, "mT"))
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(ax1, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_MH_major_kwargs():
    """Test MH_major.plot with keyword arguments passed to matplotlibs plot."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)
    try:
        fig, ax, ax1 = mh.plot(color="tab:green", linestyle=":", marker="o")
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(ax1, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_MT_io():
    """Test export of MT-object to PNG."""
    file_path = cwd.joinpath("VSM_MT.DAT")
    mt = magmeas.MT(file_path)
    output_path = cwd.joinpath("VSM_MT.png")
    try:
        mt.plot(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()
    matplotlib.pyplot.close()


def test_MT_return2():
    """Test return of matplotlib Figure and Axes objects from MH_major.plot if
    derivative==False.
    """
    file_path = cwd.joinpath("VSM_MT.DAT")
    mt = magmeas.MT(file_path)
    try:
        fig, ax = mt.plot(derivative=False)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_MT_return3():
    """Test return of matplotlib Figure and Axes objects from MH_major.plot if
    derivative==True.
    """
    file_path = cwd.joinpath("VSM_MT.DAT")
    mt = magmeas.MT(file_path)
    try:
        fig, ax, ax1 = mt.plot()
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(ax1, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_batch_io():
    """Test plot_batch with MH_major objects and saving to PNG."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)
    output_path = cwd.joinpath(file_path.stem + "_plot.png")
    try:
        magmeas.plot_batch([mh] * 3, output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()
    matplotlib.pyplot.close()


def test_batch_MH_major_return3():
    """Test return of matplotlib Figure and Axes objects from plot_batch of
    several MH_major if demag==True.
    """
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)
    try:
        fig, ax, ax1 = magmeas.plot_batch([mh] * 3)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(ax1, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_batch_MH_major_return2():
    """Test return of matplotlib Figure and Axes objects from plot_batch of
    several MH_major if demag==False.
    """
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)
    try:
        fig, ax = magmeas.plot_batch([mh] * 3, demag=False)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_batch_MH():
    """Test plot_batch with several MH-objects."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH(file_path)
    try:
        fig, ax = magmeas.plot_batch([mh] * 3)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_batch_MT_return2():
    """Test return of matplotlib Figure and Axes objects from plot_batch of
    several MT if derivative==False.
    """
    file_path = cwd.joinpath("VSM_MT.DAT")
    mt = magmeas.MT(file_path)
    try:
        fig, ax = magmeas.plot_batch([mt] * 3, derivative=False)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_batch_MT_return3():
    """Test return of matplotlib Figure and Axes objects from plot_batch of
    several MT if derivative==True.
    """
    file_path = cwd.joinpath("VSM_MT.DAT")
    mt = magmeas.MT(file_path)
    try:
        fig, ax, ax1 = magmeas.plot_batch([mt] * 3)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(ax1, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_batch_labels():
    """Test plot_batch with several MH_major objects and specified labels."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)
    try:
        fig, ax, ax1 = magmeas.plot_batch([mh] * 3, labels=["label"] * 3)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(ax1, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_batch_kwargs():
    """Test plot_major with keyword arguments passed to matplotlibs plot."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)
    try:
        fig, ax, ax1 = magmeas.plot_batch([mh] * 3, linestyle=":", marker="o")
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(ax1, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_batch_cmap_str():
    """Test plot_batch with manually specified cmap as string."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)
    try:
        fig, ax, ax1 = magmeas.plot_batch([mh] * 3, cmap="inferno")
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(ax1, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_batch_cmap_mpl():
    """Test plot_batch with manually specified cmap as matplotlib.ColorMap."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)
    try:
        fig, ax, ax1 = magmeas.plot_batch(
            [mh] * 3, cmap=matplotlib.colormaps.get_cmap("plasma")
        )
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(ax1, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_batch_AttributeError_color():
    """Test occurence of AttributeError when passing keyword argument 'color'
    to plot_batch.
    """
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)
    with pytest.raises(AttributeError):
        magmeas.plot_batch([mh] * 3, color="tab:orange")
    matplotlib.pyplot.close()


def test_batch_TypeError_cmap():
    """Test Occurence of TypeError when incorrect type is passed as cmap."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)
    with pytest.raises(TypeError):
        magmeas.plot_batch([mh] * 3, cmap=123)
    matplotlib.pyplot.close()


def test_batch_TypeError_reftype():
    """Test Occurence of TypeError when data in plot_batch contains objects of
    several different types.
    """
    file_path = cwd.joinpath("VSM_MH.DAT")
    with pytest.raises(TypeError):
        magmeas.plot_batch([magmeas.MH(file_path), magmeas.MH_major(file_path)])
    matplotlib.pyplot.close()
