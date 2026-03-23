"""Testing routine for export functionality of magmeas.VSM-derived objects."""

from pathlib import Path

import magmeas

cwd = Path(__file__).parent.resolve()


def test_VSM_yml():
    """Test export of M(H) measurement to YAML."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    data = magmeas.VSM(file_path)
    output_path = cwd.joinpath("VSM_MH.yml")
    try:
        data.to_yaml(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_VSM_csv():
    """Test export of M(H) measurement to CSV."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    data = magmeas.VSM(file_path)
    output_path = cwd.joinpath("VSM_MH.csv")
    try:
        data.to_csv(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_VSM_hdf5():
    """Test export of M(H) measurement to HDF5."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    data = magmeas.VSM(file_path)
    output_path = cwd.joinpath("VSM_MH.hdf5")
    try:
        data.to_hdf5(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_MH_major_yml():
    """Test export of M(H) measurement to YAML."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    data = magmeas.MH_major(file_path)
    output_path = cwd.joinpath("VSM_MH.yml")
    try:
        data.to_yaml(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_MH_major_csv():
    """Test export of M(H) measurement to CSV."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    data = magmeas.MH_major(file_path)
    output_path = cwd.joinpath("VSM_MH.csv")
    try:
        data.to_csv(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.with_stem(output_path.stem + "_data").is_file()
    assert output_path.with_stem(output_path.stem + "_properties").is_file()
    output_path.with_stem(output_path.stem + "_data").unlink()
    output_path.with_stem(output_path.stem + "_properties").unlink()


def test_MH_major_hdf5():
    """Test export of M(H) measurement to HDF5."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    data = magmeas.MH_major(file_path)
    output_path = cwd.joinpath("VSM_MH.hdf5")
    try:
        data.to_hdf5(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_MT_yml():
    """Test export of M(H) measurement to YAML."""
    file_path = cwd.joinpath("VSM_MT.DAT")
    data = magmeas.MT(file_path)
    output_path = cwd.joinpath("VSM_MT.yml")
    try:
        data.to_yaml(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_MT_csv():
    """Test export of M(H) measurement to CSV."""
    file_path = cwd.joinpath("VSM_MT.DAT")
    data = magmeas.MT(file_path)
    output_path = cwd.joinpath("VSM_MT.csv")
    try:
        data.to_csv(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.with_stem(output_path.stem + "_data").is_file()
    assert output_path.with_stem(output_path.stem + "_properties").is_file()
    output_path.with_stem(output_path.stem + "_data").unlink()
    output_path.with_stem(output_path.stem + "_properties").unlink()


def test_MT_hdf5():
    """Test export of M(H) measurement to HDF5."""
    file_path = cwd.joinpath("VSM_MT.DAT")
    data = magmeas.MT(file_path)
    output_path = cwd.joinpath("VSM_MT.hdf5")
    try:
        data.to_hdf5(output_path)
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
