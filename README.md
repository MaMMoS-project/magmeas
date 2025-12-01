# magmeas

Python module enabling the import and handling of VSM-data acquired on a Quantum Design PPMS measurement system.
Extrinsic permanent magnet properties can be automatically calculated from M(H) hysteresis loops and exported to YAML- or CSV-files. Basic plotting and export to HDF5 files is also available.
The same functionality is available for M(T) measurements, although greater care should be taken.
Manual inspection of the derived properties ist always encouraged.


| Description   | Badge |
|---------------|-------|
| Tests         | [![Test package](https://github.com/MaMMoS-project/magmeas/actions/workflows/test.yml/badge.svg)](https://github.com/MaMMoS-project/magmeas/actions/workflows/test.yml) |
| Releases      | [![PyPI version](https://badge.fury.io/py/magmeas.svg)](https://badge.fury.io/py/magmeas) |
| License       | [![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) |
| DOI           | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15782457.svg)](https://doi.org/10.5281/zenodo.15782457) |

## Installation

Just install with pip:

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

With m being the sample mass in mg, a and b being the sample dimensions perpendicular to the magnetization axis and
c being the sample dimension parallel to the magnetization axis. So far only cuboid samples are supported and the density
is assumed to be m / (a * b * c)

### Python

The main functionality is handled over the class magmeas.VSM and the two functions magmeas.mult_properties_to_file and plot_multiple_VSM which take VSM-objects as arguments.
Documentation is mainly available in the form of detailed docstrings. Just use `help(magmeas.VSM)` or similar to get more information. Some examples:

```python
import magmeas as mm

dat1 = mm.VSM("file1.DAT")
dat2 = mm.VSM("file2.DAT")
dat3 = mm.VSM("file3.DAT", read_method="manual", calc_properties=False)

dat1.plot("file1_plot.png")
dat1.properties_to_file("file1_properties.yaml")

mm.plot_multiple_VSM([dat1, dat2], labels=["Measurement 1", "Measurement 2"])
mm.mult_properties_to_file([dat1, dat2], "properties.txt", ["Measurement 1", "Measurement 2"])
```

### Command line

Some basic functionality is also available in the form of a command line interface. Get more information with:

```bash
magmeas -h
```


### Dependencies
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [SciPy](https://scipy.org/)
* [h5py](https://www.h5py.org/)
* [mammos-entity](https://mammos-project.github.io/mammos/index.html)
* [mammos-analysis](https://mammos-project.github.io/mammos/index.html)

Requires Python >= 3.11 due to dependencies
