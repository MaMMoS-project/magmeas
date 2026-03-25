"""Testing routine for export functionality of magmeas.VSM-derived objects."""

from pathlib import Path

import pytest

import magmeas

cwd = Path(__file__).parent.resolve()


def test_VSM_yml():
    """Test export of VSM-object to YAML."""
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
    """Test export of VSM-object to CSV."""
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
    """Test export of VSM-object to HDF5."""
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
    """Test export of MH_major-object to YAML."""
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
    """Test export of MH_major-object to CSV."""
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
    """Test export of MH_major-object to HDF5."""
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
    """Test export of MT-object to YAML."""
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
    """Test export of MT-object to CSV."""
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
    """Test export of MT-object to HDF5."""
    file_path = cwd.joinpath("VSM_MT.DAT")
    data = magmeas.MT(file_path)
    output_path = cwd.joinpath("VSM_MT.hdf5")
    try:
        data.to_hdf5(output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_batch_yml():
    """Test export of multiple MH_major-objects to YAML."""
    mh_path = cwd.joinpath("VSM_MH.DAT")
    mt_path = cwd.joinpath("VSM_MT.DAT")
    vsm = magmeas.VSM(mh_path)
    mh = magmeas.MH(mh_path)
    mhm = magmeas.MH_major(mh_path)
    mt = magmeas.MT(mt_path)
    output_path = cwd.joinpath("batch.yaml")
    try:
        magmeas.to_yaml([vsm, mh, mhm, mt], output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_batch_hdf5():
    """Test export of multiple MH_major-objects to HDF5."""
    mh_path = cwd.joinpath("VSM_MH.DAT")
    mt_path = cwd.joinpath("VSM_MT.DAT")
    vsm = magmeas.VSM(mh_path)
    mh = magmeas.MH(mh_path)
    mhm = magmeas.MH_major(mh_path)
    mt = magmeas.MT(mt_path)
    output_path = cwd.joinpath("batch.hdf5")
    try:
        magmeas.to_hdf5([vsm, mh, mhm, mt], output_path)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    assert output_path.is_file()
    output_path.unlink()


def test_batch_AttributeError_prop():
    """Test for occurence of AttributeError when argument prop_only is set to
    True while any of the objects in data do not contain any properties.
    """
    mh_path = cwd.joinpath("VSM_MH.DAT")
    mt_path = cwd.joinpath("VSM_MT.DAT")
    vsm = magmeas.VSM(mh_path)
    mh = magmeas.MH(mh_path)
    mhm = magmeas.MH_major(mh_path)
    mt = magmeas.MT(mt_path)
    with pytest.raises(AttributeError):
        magmeas.to_batch([vsm, mh, mhm, mt], prop_only=True)
