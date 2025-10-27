"""Testing routine for magmeas.VSM objects."""

import os

import magmeas


def test_version():
    """Check that __version__ exists and is a string."""
    assert isinstance(magmeas.__version__, str)


def test_init_VSM_MH():
    """Test initialisation of M(H) measurement as magmeas.VSM object."""
    example_file_path = os.path.join(os.path.dirname(__file__), "VSM_MH.DAT")
    mh = magmeas.VSM(example_file_path)
    assert isinstance(mh, magmeas.VSM)


def test_init_VSM_MT():
    """Test initialisation of M(T) measurement as magmeas.VSM object."""
    example_file_path = os.path.join(os.path.dirname(__file__), "VSM_MT.DAT")
    mh = magmeas.VSM(example_file_path)
    assert isinstance(mh, magmeas.VSM)
