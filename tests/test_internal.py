"""Testing routine for initialisation and internal calculations of
magmeas.VSM-derived objects.
"""

from pathlib import Path

import mammos_entity as me

import magmeas

cwd = Path(__file__).parent.resolve()


def test_version():
    """Check that __version__ exists and is a string."""
    assert isinstance(magmeas.__version__, str)


def test_VSM():
    """Test initialisation of M(H) measurement as magmeas.VSM object."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.VSM(file_path)
    assert isinstance(mh, magmeas.VSM)


def test_MH():
    """Test initialisation of M(H) measurement as magmeas.VSM object."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH(file_path)
    assert isinstance(mh, magmeas.MH)


def test_MH_major():
    """Test initialisation of M(H) measurement as magmeas.VSM object."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)
    assert isinstance(mh, magmeas.MH_major)


def test_calc_Ms():
    """Test initialisation of M(H) measurement as magmeas.VSM object."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)
    mh.estimate_saturation()
    mh.estimate_saturation(0.9)
    assert isinstance(mh.saturation, me.Entity)


def test_MH_major_segments():
    """Test MT.segments."""
    file_path = cwd.joinpath("VSM_MH.DAT")
    mh = magmeas.MH_major(file_path)
    assert len(mh.segments()) == 5


def test_MT():
    """Test initialisation of M(T) measurement as magmeas.VSM object."""
    file_path = cwd.joinpath("VSM_MT.DAT")
    mt = magmeas.MT(file_path)
    assert isinstance(mt, magmeas.MT)


def test_MT_segments():
    """Test MT.segments."""
    file_path = cwd.joinpath("VSM_MT.DAT")
    mt = magmeas.MT(file_path)
    assert len(mt.segments()) == 1
