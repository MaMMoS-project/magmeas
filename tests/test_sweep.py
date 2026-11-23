"Testing functionality for magmeas.Sweep class."

from pathlib import Path

import mammos_entity as me
import pytest

import magmeas

cwd = Path(__file__).parent.resolve()


def test_version():
    """Check that __version__ exists and is a string."""
    assert isinstance(magmeas.__version__, str)


def test_Sweep_init():
    """Test different initialisations of magmeas.VSM objects."""
    sweep = magmeas.Sweep()
    sweep = magmeas.Sweep().read_PPMS(cwd.joinpath("VSM_MH.DAT"))
    sweep = magmeas.Sweep(**dict(sweep))


def test_get_entity_key():
    """Test the magmeas.Sweep.get_entity_key method."""
    assert magmeas.Sweep().get_entity_key("Time") is None
    assert isinstance(
        magmeas.Sweep(t=me.Entity("Time")).get_entity_key("Time"),
        str,
    )
    assert isinstance(
        magmeas.Sweep(
            t1=me.Entity("Time"),
            t2=me.Entity("Time"),
        ).get_entity_key("Time"),
        list,
    )


def test_calc_M():
    """Test the magmeas.Sweep.calc_M method."""
    with pytest.raises(AttributeError):
        magmeas.Sweep().calc_M()
    with pytest.raises(AttributeError):
        magmeas.Sweep(
            mu1=me.Entity("MagneticMoment"),
            mu2=me.Entity("MagneticMoment"),
        ).calc_M()
    with pytest.raises(TypeError):
        magmeas.Sweep(mu=me.Entity("MagneticMoment")).calc_M(length=[1, 2, 3])
    with pytest.raises(AttributeError):
        magmeas.Sweep(mu=me.Entity("MagneticMoment")).calc_M()
