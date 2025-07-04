# magmeas

Python module that enables the import and handling of VSM-data acquired on a Quantum Design PPMS measurement system.
Extrinsic magnetic properties can be calculated from M(H) hysteresis loops and exported to JSON- or CSV-files.
Automatic plotting with matplotlib is also available.
The same functionality is available for M(T) measurements, although it can only be used for measurements
containing a single Curie-temperature and great care should be taken.

## Installation

Please clone or download this repository and install with pip.

```bash
cd magmeas
pip install .
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


If you want to store the calculated properties in a JSON-file, use "-d" or "--dump":

```bash
magmeas -d file.DAT
```


The flags can also be combined. To store the properties in a JSON-file, plot the measurement and save it as a png
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
There are also functions that plot multiple VSM objects or write the properties of several measurements to a JSON or CSV-like .TXT file.

```python
import magmeas as mm

dat1 = mm.VSM("file1.DAT")
dat2 = mm.VSM("file2.DAT", read_method="manual")

dat1.plot("file1_plot.png")
dat2.properties_to_json("file2_properties.json")

mm.plot_multiple_VSM([dat1, dat2], ["Measurement 1", "Measurement 2"])
mm.mult_properties_to_txt("properties.txt", [dat1, dat2], ["Measurement 1", "Measurement 2"])
```


## Dependencies
* NumPy
* Pandas
* Matplotlib
* h5py
* mammos-entity
* PyQT6

Requires Python >= 3.6 because f-strings are used

### Code Archive

This repository is archived on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15782457.svg)](https://doi.org/10.5281/zenodo.15782457)