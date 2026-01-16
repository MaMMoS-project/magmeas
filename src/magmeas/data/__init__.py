"""Example datasets to test and examine magmeas."""

import os
from pathlib import Path

from magmeas._base import MT, MH_major

dir_path = Path(os.path.realpath(__file__)).parent


MH_MnAl = MH_major(dir_path.joinpath("VSM_MH.DAT"))
MT_MnAl = MT(dir_path.joinpath("VSM_MT.DAT"))
