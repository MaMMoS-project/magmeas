"""Testing routine for plotting of magmeas.VSM-derived objects."""

from pathlib import Path

import mammos_units as mu
import matplotlib
import pytest

import magmeas

matplotlib.use("Agg")
cwd = Path(__file__).parent.resolve()


def test_MH_io():
    """Test export of M(H) measurement to PNG."""
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
    """Test export of M(H) measurement to PNG."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH(file_path)
    try:
        fig, ax = mh.plot()
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_MH_label():
    """Test export of M(H) measurement to PNG."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH(file_path)
    try:
        fig, ax = mh.plot(label="Test_label")
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_MH_unit_str():
    """Test export of M(H) measurement to PNG."""
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
    """Test export of M(H) measurement to PNG."""
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
    """Test export of M(H) measurement to PNG."""
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
    """Test export of M(H) measurement to PNG."""
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
    """Test export of M(H) measurement to PNG."""
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
    """Test export of M(H) measurement to PNG."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH(file_path)
    with pytest.raises(ValueError):
        mh.plot(unit="kg")
    matplotlib.pyplot.close()


def test_MH_ValueError_unit_tuple_len():
    """Test export of M(H) measurement to PNG."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH(file_path)
    with pytest.raises(ValueError):
        mh.plot(unit=("T", "T", "T"))
    matplotlib.pyplot.close()


def test_MH_major_io():
    """Test export of M(H) measurement to PNG."""
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
    """Test export of M(H) measurement to PNG."""
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
    """Test export of M(H) measurement to PNG."""
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


def test_MH_major_kwargs():
    """Test export of M(H) measurement to PNG."""
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


def test_MH_major_unit():
    """Test export of M(H) measurement to PNG."""
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


def test_MT_io():
    """Test export of M(T) measurement to PNG."""
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


def test_MT_return3():
    """Test export of M(T) measurement to PNG."""
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


def test_MT_return2():
    """Test export of M(T) measurement to PNG."""
    file_path = cwd.joinpath("VSM_MT.DAT")
    mt = magmeas.MT(file_path)
    try:
        fig, ax = mt.plot(derivative=False)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_plot_mult_MH_major_io():
    """Test export of multiple M(H) measurement to YAML."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh1 = magmeas.MH_major(file_path)
    mh2 = magmeas.MH_major(file_path)

    output_path = cwd.joinpath(file_path.stem + "_plot.png")
    try:
        magmeas.plot_multiple_MH_major([mh1, mh2], output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()
    matplotlib.pyplot.close()


def test_plot_mult_MH_major_return3():
    """Test export of multiple M(H) measurement to YAML."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh1 = magmeas.MH_major(file_path)
    mh2 = magmeas.MH_major(file_path)
    try:
        fig, ax, ax1 = magmeas.plot_multiple_MH_major([mh1, mh2])
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(ax1, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_plot_mult_MH_major_return2():
    """Test export of multiple M(H) measurement to YAML."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh1 = magmeas.MH_major(file_path)
    mh2 = magmeas.MH_major(file_path)
    try:
        fig, ax = magmeas.plot_multiple_MH_major([mh1, mh2], demag=False)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_plot_mult_MH_major_labels():
    """Test export of multiple M(H) measurement to YAML."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh1 = magmeas.MH_major(file_path)
    mh2 = magmeas.MH_major(file_path)
    try:
        fig, ax, ax1 = magmeas.plot_multiple_MH_major(
            [mh1, mh2], labels=["label1", "label2"]
        )
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(ax1, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_plot_mult_MH_major_kwargs():
    """Test export of multiple M(H) measurement to YAML."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh1 = magmeas.MH_major(file_path)
    mh2 = magmeas.MH_major(file_path)
    try:
        fig, ax, ax1 = magmeas.plot_multiple_MH_major(
            [mh1, mh2], linestyle=":", marker="o"
        )
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(ax1, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_plot_mult_MH_major_cmap_str():
    """Test export of multiple M(H) measurement to YAML."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh1 = magmeas.MH_major(file_path)
    mh2 = magmeas.MH_major(file_path)
    try:
        fig, ax, ax1 = magmeas.plot_multiple_MH_major([mh1, mh2], cmap="inferno")
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(ax1, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_plot_mult_MH_major_cmap_mpl():
    """Test export of multiple M(H) measurement to YAML."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh1 = magmeas.MH_major(file_path)
    mh2 = magmeas.MH_major(file_path)
    try:
        fig, ax, ax1 = magmeas.plot_multiple_MH_major(
            [mh1, mh2], cmap=matplotlib.colormaps.get_cmap("plasma")
        )
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(ax1, matplotlib.axes.Axes)
    matplotlib.pyplot.close()


def test_plot_mult_MH_major_AttributeError_color():
    """Test export of multiple M(H) measurement to YAML."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh1 = magmeas.MH_major(file_path)
    mh2 = magmeas.MH_major(file_path)
    with pytest.raises(AttributeError):
        magmeas.plot_multiple_MH_major([mh1, mh2], color="tab:orange")
    matplotlib.pyplot.close()


def test_plot_mult_MH_major_AttributeError_cmap():
    """Test export of multiple M(H) measurement to YAML."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh1 = magmeas.MH_major(file_path)
    mh2 = magmeas.MH_major(file_path)
    with pytest.raises(TypeError):
        magmeas.plot_multiple_MH_major([mh1, mh2], cmap=123)
    matplotlib.pyplot.close()
