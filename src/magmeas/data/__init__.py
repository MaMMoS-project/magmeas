"""Example datasets to test and examine magmeas."""

import os
from pathlib import Path

from magmeas.magmeas import VSM

dir_path = Path(os.path.realpath(__file__)).parent


MH_MnAl = VSM(dir_path.joinpath("VSM_MH.DAT"))
MT_MnAl = VSM(dir_path.joinpath("VSM_MT.DAT"))
