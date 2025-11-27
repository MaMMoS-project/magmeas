"""Testing routine for magmeas.VSM objects."""

from pathlib import Path

import matplotlib

import magmeas

matplotlib.use("Agg")
cwd = Path(__file__).parent.resolve()


def test_version():
    """Check that __version__ exists and is a string."""
    assert isinstance(magmeas.__version__, str)


def test_init_VSM_MH():
    """Test initialisation of M(H) measurement as magmeas.VSM object."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.VSM(file_path)
    assert isinstance(mh, magmeas.VSM)


def test_init_VSM_MT():
    """Test initialisation of M(T) measurement as magmeas.VSM object."""
    file_path = cwd.joinpath("VSM_MT.DAT")
    mt = magmeas.VSM(file_path)
    assert isinstance(mt, magmeas.VSM)


def test_VSM_MH_to_yml():
    """Test export of M(H) measurement to YAML."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.VSM(file_path)
    output_path = cwd.joinpath(file_path.stem + "_properties.yml")
    try:
        mh.properties_to_file(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_VSM_MH_to_csv():
    """Test export of M(H) measurement to CSV."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.VSM(file_path)
    output_path = cwd.joinpath(file_path.stem + "_properties.csv")
    try:
        mh.properties_to_file(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_VSM_MH_to_hdf5():
    """Test export of M(H) measurement to HDF5."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.VSM(file_path)
    output_path = cwd.joinpath(file_path.stem + ".hdf5")
    try:
        mh.to_hdf5()
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_VSM_MH_plot():
    """Test export of M(H) measurement to PNG."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.VSM(file_path)
    output_path = cwd.joinpath(file_path.stem + ".png")
    try:
        mh.plot(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_VSM_MT_to_yml():
    """Test export of M(T) measurement to YAML."""
    file_path = cwd.joinpath("VSM_MT.DAT")
    mh = magmeas.VSM(file_path)
    output_path = cwd.joinpath(file_path.stem + "_properties.yml")
    try:
        mh.properties_to_file(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_VSM_MT_to_csv():
    """Test export of M(T) measurement to CSV."""
    file_path = cwd.joinpath("VSM_MT.DAT")
    mh = magmeas.VSM(file_path)
    output_path = cwd.joinpath(file_path.stem + "_properties.csv")
    try:
        mh.properties_to_file(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_VSM_MT_to_hdf5():
    """Test export of M(T) measurement to HDF5."""
    file_path = cwd.joinpath("VSM_MT.DAT")
    mh = magmeas.VSM(file_path)
    output_path = cwd.joinpath(file_path.stem + ".hdf5")
    try:
        mh.to_hdf5()
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_VSM_MT_plot():
    """Test export of M(T) measurement to PNG."""
    file_path = cwd.joinpath("VSM_MT.DAT")
    mh = magmeas.VSM(file_path)
    output_path = cwd.joinpath(file_path.stem + ".png")
    try:
        mh.plot(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()
