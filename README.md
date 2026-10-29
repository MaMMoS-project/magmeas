# magmeas

Python module enabling the import and analysis of **mag**netic **meas**urements.


| Description   | Badge |
|---------------|-------|
| Documentation | [![Documentation](https://img.shields.io/badge/Docs-mammos--project.github.io%2Fmagmeas-blue)](https://mammos-project.github.io/magmeas/) |
| Tests         | [![Test package](https://github.com/MaMMoS-project/magmeas/actions/workflows/test.yml/badge.svg)](https://github.com/MaMMoS-project/magmeas/actions/workflows/test.yml) |
| Releases      | [![PyPI version](https://badge.fury.io/py/magmeas.svg)](https://badge.fury.io/py/magmeas) |
| License       | [![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) |
| DOI           | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15782456.svg)](https://doi.org/10.5281/zenodo.15782456) |

## Installation

Just install from PyPI:

```bash
pip install magmeas
```

## Usage

There are three distinct classes that conveniently handle specific types of measurement data:
* `Sweep`: general for all types of magnetometry measurements
* `MH_major`: for major hysteresis loop measurements
* `MT`: for M(T) measurements

Here is what each of them do at initialisation or automatically with a method:

|  | `Sweep` | `MH_major` | `MT` |
|--|---------|------------|------|
|accept general measurement data | ✅ | ✅ | ✅ |
|import measurement data from Quantum Design .DAT file | ✅ | ✅ | ✅ |
|export measurement data to csv/yml/hdf5 | ✅ | ✅ | ✅ |
|plotting | ✅ | ✅ | ✅ |
|calculate properties | ❌ | ✅ $H_c$, $M_r$, $BH_{max}$, $H_k$, S, $M_s$ (optional) | ✅ $T_c$ |

Experimental support for Recoil-loop- and FORC-measurements is also provided in their respective classes.
All classes inherit from the `EntityCollection` class in [`mammos-entity`](https://github.com/MaMMoS-project/mammos-entity), using its io functionality. In doing so, they can conveniently be added to a new `EntityCollection`, benefitting from the nested structure in these objects even for data export.
Plotting of the above highlighted classes is very flexible. Common keyword arguments used in matplotlibs functions can be used. Sweep-derived objects can either be plotted to pre-existing matplotlib figures and axes, or they dynamically generate new ones and return them for subsequent formatting.

Requires Python >= 3.11, < 3.15 due to dependencies
