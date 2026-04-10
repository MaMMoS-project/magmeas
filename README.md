# magmeas

Python module enabling the import and handling of VSM-data acquired on a Quantum Design PPMS/DynaCool/Versalab measurement system.


| Description   | Badge |
|---------------|-------|
| Documentation | [![Documentation](https://img.shields.io/badge/Docs-mammos--project.github.io%2Fmagmeas-blue)](https://mammos-project.github.io/magmeas/) |
| Tests         | [![Test package](https://github.com/MaMMoS-project/magmeas/actions/workflows/test.yml/badge.svg)](https://github.com/MaMMoS-project/magmeas/actions/workflows/test.yml) |
| Releases      | [![PyPI version](https://badge.fury.io/py/magmeas.svg)](https://badge.fury.io/py/magmeas) |
| License       | [![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) |
| DOI           | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15782457.svg)](https://doi.org/10.5281/zenodo.15782456) |

## Installation

Just install from PyPI:

```bash
pip install magmeas
```

## Usage

### Input file formatting

For conveniently importing automatically, make sure that the necessary sample properties are given in the .DAT file. The formatting is as follows:

```
INFO,m,SAMPLE_MASS
INFO,(a, b, c),SAMPLE_SIZE
```

With `m` being the sample mass in mg, `a` and `b` being the sample dimensions perpendicular to the magnetization axis and
`c` being the sample dimension parallel to the magnetization axis. So far only cuboid samples are supported and the density
is assumed to be m / (a * b * c)

### Python

There are four distinct classes that conveniently handle specific types of measurement data:
* `VSM`: general for all types of VSM measurements
* `MH`: for all M(H) measurements, still quite unspecific
* `MH_major`: for major hysteresis loop measurements
* `MT`: for M(T) measurements

Here is what each of them do at initialisation or automatically with a method:

|  | `VSM` | `MH` | `MH_major` | `MT` |
|--|-----|----|----------|----|
| import measurement data from Quantum Design .DAT file | ✅ | ✅ | ✅ | ✅ |
|export measurement data to csv/yml/hdf5 | ✅ | ✅ | ✅ | ✅ |
| plotting | ❌ | ✅ | ✅ | ✅ |
| calculate properties | ❌ | ❌ | ✅ $H_c$, $M_r$, $BH_{max}$, $H_k$, S, $M_s$ (optional) | ✅ $T_c$ |
|export properties to csv/yml/hdf5 | ❌ | ❌ | ✅ | ✅ |

Experimental support for Recoil-loop- and FORC-measurements is also provided in their respective classes.
All classes inherit from the `EntityCollection` class in [`mammos-entity`](https://github.com/MaMMoS-project/mammos-entity), using its io functionality. In doing so, they can conveniently be added to a new `EntityCollection` with `magmeas.to_batch`, benefitting from the nested structure in these objects even for data export.
Plotting of the above highlighted classes is very flexible. Common keyword arguments used in matplotlibs functions can be used. VSM-derived objects can either be plotted to pre-existing matplotlib figures and axes, or they dynamically generate new ones and return them for subsequent formatting.

Documentation is mainly available in the form of detailed docstrings. Just use `dir(magmeas)` and `help(magmeas.VSM)` or similar to get more information.

Requires Python >= 3.11 due to dependencies
