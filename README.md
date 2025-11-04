# magmeas

Python module that enables the import and handling of VSM-data acquired on a Quantum Design PPMS measurement system.
Extrinsic magnetic properties can be calculated from M(H) hysteresis loops and exported to YAML- or CSV-files.
The automatic handling of M(H)-measurements is mainly optimised for permanent magnets.
Automatic plotting with matplotlib is also available.
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

Make sure that the necessary sample properties are given in the .DAT file. The formatting is as follows:

```
INFO,m,SAMPLE_MASS
INFO,(a, b, c),SAMPLE_SIZE
```

With m being the sample mass in mg, a and b being the sample dimensions perpendicular to the magnetization axis and
c being the sample dimension parallel to the magnetization axis. So far only cuboid samples are supported and the density
is assumed to be m / (a * b * c)

### Command line

Get a quick overview with the "--help" or "-h" flag

```bash
magmeas -h
```


To print the magnetic properties calculated from the measurement saved in "file.DAT" to the console you can use:

```bash
magmeas file.DAT
```


To plot the measurement with matplotlib, type "-p" or "--plot":

```bash
magmeas -p file.DAT
```


If you want to store the calculated properties in a YAML-file, use "-d" or "--dump":

```bash
magmeas -d file.DAT
```


The flags can also be combined. To store the properties in a YAML-file, plot the measurement and save it as a png
and prevent the printing of any information to the console with the flag "-s" or "--silent":

```bash
magmeas -dps file.DAT
```


To operate on all files in a folder, just give the path to the folder in question:

```bash
magmeas -d folder_with_files
```


The content of the specified DAT-file can also be written into an hdf5-file together with the calculated properties

```bash
magmeas --hdf5 file.DAT
```


### Python

You can also import the module in python and use it with a bit more control. Most things are handled within the class "VSM".
This reads in .DAT files which are formatted as specified above or those which had their properties entered manually in a console dialog.
There are also functions that plot multiple VSM objects or write the properties of several measurements to a YAML or CSV-like .TXT file.

```python
import magmeas as mm

dat1 = mm.VSM("file1.DAT")
dat2 = mm.VSM("file2.DAT", read_method="manual")

dat1.plot("file1_plot.png")
dat2.properties_to_file("file2_properties.yaml")

mm.plot_multiple_VSM([dat1, dat2], labels=["Measurement 1", "Measurement 2"])
mm.mult_properties_to_file([dat1, dat2], "properties.txt", ["Measurement 1", "Measurement 2"])
```


## Dependencies
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [SciPy](https://scipy.org/)
* [h5py](https://www.h5py.org/)
* [mammos-entity](https://mammos-project.github.io/mammos/index.html)
* [mammos-analysis](https://mammos-project.github.io/mammos/index.html)

Requires Python >= 3.11 due to dependencies
