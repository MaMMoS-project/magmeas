"""Testing routine for magmeas.VSM objects."""

from pathlib import Path

import mammos_entity as me
import matplotlib

import magmeas

matplotlib.use("Agg")
cwd = Path(__file__).parent.resolve()


def test_version():
    """Check that __version__ exists and is a string."""
    assert isinstance(magmeas.__version__, str)


def test_init_MH_major():
    """Test initialisation of M(H) measurement as magmeas.VSM object."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)

    assert isinstance(mh, magmeas.MH_major)


def test_init_MT():
    """Test initialisation of M(T) measurement as magmeas.VSM object."""
    file_path = cwd.joinpath("VSM_MT.DAT")
    mt = magmeas.MT(file_path)

    assert isinstance(mt, magmeas.MT)


def test_calc_Ms():
    """Test initialisation of M(H) measurement as magmeas.VSM object."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)

    mh.estimate_saturation()
    mh.estimate_saturation(0.9)
    assert isinstance(mh.saturation, me.Entity)
    assert isinstance(mh.properties["Ms"], me.Entity)


def test_MH_major_to_yml():
    """Test export of M(H) measurement to YAML."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)

    file_path = cwd.joinpath("VSM_MH.DAT")
    output_path = cwd.joinpath(file_path.stem + "_properties.yml")
    try:
        mh.properties_to_file(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_MH_major_to_csv():
    """Test export of M(H) measurement to CSV."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)

    file_path = cwd.joinpath("VSM_MH.DAT")
    output_path = cwd.joinpath(file_path.stem + "_properties.csv")
    try:
        mh.properties_to_file(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_MH_major_to_hdf5():
    """Test export of M(H) measurement to HDF5."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)

    file_path = cwd.joinpath("VSM_MH.hdf5")
    try:
        mh.to_hdf5(file_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert file_path.is_file()
    file_path.unlink()


def test_MH_major_plot():
    """Test export of M(H) measurement to PNG."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)

    file_path = cwd.joinpath("VSM_MH.DAT")
    output_path = cwd.joinpath(file_path.stem + ".png")
    try:
        mh.plot(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_MT_to_yml():
    """Test export of M(T) measurement to YAML."""
    file_path = cwd.joinpath("VSM_MT.DAT")
    mt = magmeas.MT(file_path)

    file_path = cwd.joinpath("VSM_MT.DAT")
    output_path = cwd.joinpath(file_path.stem + "_properties.yml")
    try:
        mt.properties_to_file(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_MT_to_csv():
    """Test export of M(T) measurement to CSV."""
    file_path = cwd.joinpath("VSM_MT.DAT")
    mt = magmeas.MT(file_path)

    file_path = cwd.joinpath("VSM_MT.DAT")
    output_path = cwd.joinpath(file_path.stem + "_properties.csv")
    try:
        mt.properties_to_file(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_MT_to_hdf5():
    """Test export of M(T) measurement to HDF5."""
    file_path = cwd.joinpath("VSM_MT.DAT")
    mt = magmeas.MT(file_path)

    file_path = cwd.joinpath("VSM_MT.hdf5")
    try:
        mt.to_hdf5(file_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert file_path.is_file()
    file_path.unlink()


def test_MT_plot():
    """Test export of M(T) measurement to PNG."""
    file_path = cwd.joinpath("VSM_MT.DAT")
    mt = magmeas.MT(file_path)

    file_path = cwd.joinpath("VSM_MT.DAT")
    output_path = cwd.joinpath(file_path.stem + ".png")
    try:
        mt.plot(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_mult_MH_major_to_yml():
    """Test export of multiple M(H) measurement to YAML."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh1 = magmeas.MH_major(file_path)
    mh2 = magmeas.MH_major(file_path)

    output_path = cwd.joinpath(file_path.stem + "_properties.yml")
    try:
        magmeas.mult_properties_to_file([mh1, mh2], output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_plot_mult_MH_major():
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
