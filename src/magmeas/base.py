"""VSM class and functions using it."""

from importlib.metadata import version
from pathlib import Path

import h5py
import mammos_entity as me
import mammos_units as mu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mammos_analysis.hysteresis import extrinsic_properties
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress

from magmeas._utils import rextract

mu.set_enabled_equivalencies(mu.magnetic_flux_field())


class VSM:
    """
    Class for importing, storing and using of VSM-data aswell as derived
    parameters.

    Attributes
    ----------
    H: ENTITY
        Internal magnetic field as mammos_entity.Entity
    H_ext: ENTITY
        External magnetic field as mammos_entity.Entity
    M: ENTITY
        Magnetization as mammos_entity.Entity
    m: Quantity
        Magnetic Moment as mammos_units.Quantity
    T: ENTITY
        Absolute temperature as mammos_entity.Entity
    t: ENTITY
        Time as mammos_entity.Entity
    D: ENTITY
        Demagnetizing factor as mammos_entity.Entity

    Methods
    -------
    to_hdf5()
        Save contents of .DAT-file and calculated properties in hdf5 file.
    """

    def __init__(self, datfile, read_method="auto"):
        # import data
        self.load_qd(datfile, read_method=read_method)

    def _demag_prism(self, dim):
        r"""
        Calculate demagnetization factor Dz for a rectangular prism.
        Dimensions are a, b, and c. c is assumed to be the axis along which
        the prism was magnetized.
        Copied from
        https://rmlmcfadden.github.io/bnmr/technical-information/calculators/
        Equation from A. Aharoni, J. Appl. Phys. 83, 3422 (1998).
        https://doi.org/10.1063/1.367113
        Eq. (1) - see Fig. 1 for_abc coordinate system.

        Parameters
        ----------
        dim: LIST | ARRAY
            List of sample dimensions as mammos_units.Quantity values

        Returns
        -------
        D: ENTITY
            Demagnetization factor along axis of magnetization (c-axis) as
            mammos_entity.Entity
        """
        # the expression takes input as half of the semi-axes
        a = 0.5 * dim[0]
        b = 0.5 * dim[1]
        c = 0.5 * dim[2]  # c is || axis along which the prism was magnetized
        # define some convenience terms
        a2 = a * a
        b2 = b * b
        c2 = c * c
        abc = a * b * c
        ab = a * b
        ac = a * c
        bc = b * c
        r_abc = np.sqrt(a2 + b2 + c2)
        r_ab = np.sqrt(a2 + b2)
        r_bc = np.sqrt(b2 + c2)
        r_ac = np.sqrt(a2 + c2)
        # compute the factor
        pi_Dz = (
            ((b2 - c2) / (2 * bc)) * np.log((r_abc - a) / (r_abc + a))
            + ((a2 - c2) / (2 * ac)) * np.log((r_abc - b) / (r_abc + b))
            + (b / (2 * c)) * np.log((r_ab + a) / (r_ab - a))
            + (a / (2 * c)) * np.log((r_ab + b) / (r_ab - b))
            + (c / (2 * a)) * np.log((r_bc - b) / (r_bc + b))
            + (c / (2 * b)) * np.log((r_ac - a) / (r_ac + a))
            + 2 * np.arctan2(ab, c * r_abc) / mu.rad
            + (a2 * a + b2 * b - 2 * c2 * c) / (3 * abc)
            + ((a2 + b2 - 2 * c2) / (3 * abc)) * r_abc
            + (c / ab) * (r_ac + r_bc)
            - (r_ab * r_ab * r_ab + r_bc * r_bc * r_bc + r_ac * r_ac * r_ac) / (3 * abc)
        )
        # divide out the factor of pi
        D = pi_Dz / np.pi
        return me.Entity("DemagnetizingFactor", D.value)

    def load_qd(self, datfile, read_method):
        """
        Load VSM-data from a quantum systems .DAT file.

        Parameters
        ----------
        datfile: STR | PATH
            Path to quantum systems .DAT file that data is supposed to be
            imported from

        read_method: STR
            Determines whether magmeas will attempt to automatically read the
            sample parameters necessary for the following calculations or not.
            Can be "auto" or "manual"

        Returns
        -------
        None

        """
        self.path = Path(datfile)

        err = """
              Sample parameters could not be read automatically.
              Please enter sample parameters manually or enter them
              correctly in the .DAT file like this:
              INFO,<mass in mg>,SAMPLE_MASS
              INFO,(<a>, <b>, <c>),SAMPLE_SIZE
              sample dimensions a, b and c in mm, c parallel to field
              """
        # open file and extract content as string
        with open(self.path) as f:
            s = f.read()
        # figure out in which line actual CSV-like data starts for later import
        head_length = s[: s.find("[Data]")].count("\n") + 1

        # Automatically read out sample parameters
        if read_method == "auto":
            # check if sample mass can be read from .DAT file
            try:
                mass = float(rextract(s, "INFO,", ",SAMPLE_MASS")) * mu.mg
            except ValueError:
                raise Exception(err) from None
            # check if sample dimensions can be read from .DAT file
            try:
                dim = rextract(s, "INFO,(", "),SAMPLE_SIZE").split(",")
                dim = np.array([float(f) for f in dim]) * mu.mm
            except ValueError:
                raise Exception(err) from None

        # Input sample parameters manually
        elif read_method == "manual":
            print("Manual input method selected")
            mass = float(input("Sample mass in mg: ")) * mu.mg
            a = float(input("Sample dimension a (perpendicular to field) in mm: "))
            b = float(input("Sample dimension b (perpendicular to field) in mm: "))
            c = float(input("Sample dimension c (parallel to field) in mm: "))
            dim = np.array([a, b, c]) * mu.mm
        else:
            raise Exception(err)

        # calculate density from sample mass and dimensions
        density = mass / (np.prod(dim.value) * dim.unit**3)
        # calculate demagnetisation factor
        self.D = self._demag_prism(dim)
        # import measurement data
        df = pd.read_csv(self.path, skiprows=head_length, encoding="cp1252")
        # extract magnetic moment
        # first, let's find the unit of the magnetic moment
        for col in df.columns:
            start_idx = col.find("Moment (")
            if start_idx != -1:
                start_idx = col.find("(") + 1
                end_idx = col.find(")")
                m_unit = col[start_idx:end_idx]
        # if the unit is declared as 'emu', we need to specify it explicitly
        # since emu is ambiguous and thus not included in mammos-units
        if m_unit == "emu":
            m = np.array(df["Moment (emu)"]) * mu.erg / mu.G
        elif m_unit == "Am2":
            m = np.array(df["Moment (Am2)"]) * mu.A * mu.m**2
        # So far this is only a Quantity and no Entity because the wrong unit
        # seems to be defined for the me.Entity('MagneticMoment')
        else:
            raise ValueError(
                "Unit of magnetic moment seems to be neither emu,"
                " nor Am2. Other units are not supported."
            )

        # extract external magnetic field
        eH = me.Entity(
            "ExternalMagneticField", np.array(df["Magnetic Field (Oe)"]) * mu.Oe
        )
        # calculate magnetization
        M = me.Entity("Magnetization", m / mass * density)
        # calculate internal magnetic field
        H = me.Entity("InternalMagneticField", eH.q - self.D.q * M.q)
        # extract absolute temperature
        T = me.Entity("Temperature", np.array(df["Temperature (K)"]) * mu.K)
        # extract time stamp
        t = me.Entity("Time", np.array(df["Time Stamp (sec)"]) * mu.s)

        # test datapoints for missing values (where value is nan)
        # H is derived from H_ext, and m so this is sufficient to test both
        nanfilter = ~np.isnan(H.q) * ~np.isnan(T.q) * ~np.isnan(t.q)
        # delete all datapoints where H_ext, m, T or t are nan and assign them
        self.H = me.Entity(H.ontology_label, H.q.to("A/m")[nanfilter])
        self.H_ext = me.Entity(eH.ontology_label, eH.q.to("A/m")[nanfilter])
        self.M = me.Entity(M.ontology_label, M.q.to("A/m")[nanfilter])
        self.m = mu.Quantity(m.to("A m2")[nanfilter])
        self.T = me.Entity(T.ontology_label, T.q[nanfilter])
        # convert time stamp to time since measurement start
        self.t = me.Entity(t.ontology_label, t.q[nanfilter] - t.q[nanfilter][0])
        # collect all measurement data in one dictionary
        self.measurement_data = {
            "H": self.H,
            "H_ext": self.H_ext,
            "M": self.M,
            "m": self.m,
            "T": self.T,
            "t": self.t,
        }

    def __repr__(self):
        """
        Represent VSM object more helpfully.

        Parameters
        ----------
        NONE

        Returns
        -------
        NONE
        """
        module = self.__module__.split(".")[0]
        vsm_type = str(type(self)).split(".")[-1][: str(type(self)).find("'") + 1]
        return f"{module}.{vsm_type}({self.path.name})"

    def to_hdf5(self, file_path):
        """
        Save contents of .DAT-file and calculated properties in hdf5 file.

        Parameters
        ----------
        file_path: STR | PATH
            File path the HDF5-file is going to be saved under.

        Returns
        -------
        None.

        """
        # read header of .DAT file
        with open(self.path, encoding="cp1252") as f:
            s = str(f.read(-1))
        head = s[s.index("INFO") : s.rindex("\nDATATYPE")]
        head = head.split("\n")
        info = {
            i[i.rindex(",") + 1 :]: i[i.index(",") + 1 : i.rindex(",")] for i in head
        }

        # read columns of .DAT file as csv
        df = pd.read_csv(self.path, skiprows=34, encoding="cp1252")

        #  write HDF5
        file_path = Path(file_path)
        with h5py.File(file_path, "a") as f:
            # write info (header of .DAT) to HDF5
            for i in info:
                f.create_dataset("Info/" + i, data=info[i])
            f["Info/SAMPLE_MASS"].attrs["unit"] = "mg"
            f["Info/SAMPLE_SIZE"].attrs["unit"] = "mm"

            # write raw data (columns of .DAT) to HDF5
            for i in df.columns:
                dat = np.array(df[i])
                if dat.dtype != "O":
                    f.create_dataset("RawData/" + i, data=dat)
                if dat.dtype == "O":
                    f.create_dataset("RawData/" + i, data=[str(j) for j in df[i]])

            # write derived measurement data (from VSM object) to HDF5
            for key in self.measurement_data:
                if isinstance(self.measurement_data[key], me.Entity):
                    value = self.measurement_data[key].value
                    unit = str(self.measurement_data[key].unit)
                    iri = self.measurement_data[key].ontology.iri
                    f.create_dataset(f"DerivedData/{key}", data=value)
                    f[f"DerivedData/{key}"].attrs["unit"] = unit
                    f[f"DerivedData/{key}"].attrs["IRI"] = iri

                elif isinstance(self.measurement_data[key], mu.Quantity):
                    value = self.measurement_data[key].value
                    unit = str(self.measurement_data[key].unit)
                    f.create_dataset(f"DerivedData/{key}", data=value)
                    f[f"DerivedData/{key}"].attrs["unit"] = unit

                else:
                    value = self.measurement_data[key].value
                    f.create_dataset(f"DerivedData/{key}", data=value)


class MH(VSM):
    """
    Class for importing, storing and using of VSM-data from M(H) measurements.

    Attributes
    ----------
    H: ENTITY
        Internal magnetic field as mammos_entity.Entity
    H_ext: ENTITY
        External magnetic field as mammos_entity.Entity
    M: ENTITY
        Magnetization as mammos_entity.Entity
    m: Quantity
        Magnetic Moment as mammos_units.Quantity
    T: ENTITY
        Absolute temperature as mammos_entity.Entity
    t: ENTITY
        Time as mammos_entity.Entity
    D: ENTITY
        Demagnetizing factor as mammos_entity.Entity

    Methods
    -------
    plot()
        Plot data according to measurement type, optionally saves as png.
    to_hdf5()
        Save contents of .DAT-file to hdf5 file.
    """

    def plot(self, filepath=None, label=None):
        """
        Plot M(H) measurement and save figure if a filepath is given.

        Parameters
        ----------
        filepath: STR | PATH, optional
            Filepath for saving the figure. Default is None, in that case no
            file is saved.
        label: STR, optional
            Optional label of M(H)-measurement that can be displayed as title.
            Default is None, in that case no legend is displayed.

        Returns
        -------
        None
        """
        H = self.H.q.to("T")  # converts H from A/m to Tesla
        M = self.M.q.to("T")  # converts M from A/m to Tesla

        fig, ax1 = plt.subplots(1, 1, figsize=(16 / 2.54, 12 / 2.54))
        ax1.plot(H, M)

        # format plot
        ax1.set_xlabel(r"$\mu_0 H_{int}$ in $T$")
        ax1.set_ylabel(r"$J$ in $T$")
        if label is not None:
            ax1.set_title(label)

        # save figure if filepath is given
        if filepath is not None:
            fig.savefig(filepath, dpi=300)


class _Property_Container:
    """Abstract parent class that introduces handling of properties."""

    def properties_to_file(self, filepath, label=None):
        r"""
        Save all properties derived from the major hysteresis loop
        to CSV-file or to YAML file, using the mammos-entity io functionality.

        Parameters
        ----------
        filepath: STR | PATH
            Filepyth to save the file to. Ending also determines type of file.

        Returns
        -------
        None
        """
        description = (
            f"mammos-entity version = {version('mammos-entity')}\n"
            + f"magmeas       version = {version('magmeas')}"
        )

        me.io.entities_to_file(filepath, description, **self.properties)

    def print_properties(self):
        """
        Print out properties of VSM object.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        print(f"\n\n{self.path.name}:\n")

        for key in self.properties:
            print(f"{key} = {self.properties[key]}")

    def to_hdf5(self, file_path):
        """
        Save contents of .DAT-file and calculated properties in hdf5 file.

        Parameters
        ----------
        file_path: STR | PATH
            File path the HDF5-file is going to be saved under.

        Returns
        -------
        None.

        """
        # write contents of .DAT file and derived measurement data to HDF5
        super().to_hdf5(file_path)

        # write properties to HDF5
        with h5py.File(file_path, "a") as f:
            for key in self.properties:
                if isinstance(self.properties[key], me.Entity):
                    value = self.properties[key].value
                    unit = str(self.properties[key].unit)
                    iri = self.properties[key].ontology.iri
                    f.create_dataset(f"Properties/{key}", data=value)
                    f[f"Properties/{key}"].attrs["unit"] = unit
                    f[f"Properties/{key}"].attrs["IRI"] = iri

                elif isinstance(self.properties[key], mu.Quantity):
                    value = self.properties[key].value
                    unit = str(self.properties[key].unit)
                    f.create_dataset(f"Properties/{key}", data=value)
                    f[f"Properties/{key}"].attrs["unit"] = unit

                else:
                    value = self.properties[key].value
                    f.create_dataset(f"Properties/{key}", data=value)


class MH_major(MH, _Property_Container):
    """
    Class for importing, storing and using of VSM-data from major loop M(H)
    measurements aswell as derived properties.

    Attributes
    ----------
    H: ENTITY
        Internal magnetic field as mammos_entity.Entity
    H_ext: ENTITY
        External magnetic field as mammos_entity.Entity
    M: ENTITY
        Magnetization as mammos_entity.Entity
    m: Quantity
        Magnetic Moment as mammos_units.Quantity
    T: ENTITY
        Absolute temperature as mammos_entity.Entity
    t: ENTITY
        Time as mammos_entity.Entity
    D: ENTITY
        Demagnetizing factor as mammos_entity.Entity
    remanence: ENTITY
        Remanent magnetization as mammos_entity.Entity
    coercivity: ENTITY
        Internal coercive field as mammos_entity.Entity
    BHmax: ENTITY
        Maximum energy product as mammos_entity.Entity
    kneefield: ENTITY
        Knee field as mammos_entity.Entity, see ontology
    squareness: FLOAT
        Squareness as kneefield / coercivity

    Methods
    -------
    segments()
        Find indices of segmentation points which can be used to seperate each
        measurement segment from each other.
    estimate_saturation()
        Calculate an estimation of the saturation magnetisation, return it and
        simultaneously assign it to the VSM object. A possible high field
        susceptibility can optionally be corrected for.
    plot()
        Plot data according to measurement type, optionally saves as png.
    properties_to_file()
        Save all properties derived from the VSM-measurement to CSV-file or to
        YAML file, using the mammos-entity io functionality.
    to_hdf5()
        Save contents of .DAT-file and calculated properties in hdf5 file.
    """

    def __init__(self, datfile, read_method="auto"):
        # import data
        super().load_qd(datfile, read_method=read_method)

        # calculate properties
        s = self.segments()
        idx = np.argsort(self.H.q[s[0] : s[1]]) + s[0]
        prop = extrinsic_properties(self.H.q[idx], self.M.q[idx], 0)

        self.remanence = prop.Mr
        self.coercivity = prop.Hc
        self.BHmax = prop.BHmax

        self.kneefield = self._calc_kneefield()
        self.squareness = self._calc_squareness()

        self.properties = {
            "Mr": self.remanence,
            "Hc": self.coercivity,
            "BHmax": self.BHmax,
            "Hk": self.kneefield,
            "S": self.squareness,
        }

    def _calc_kneefield(self):
        """
        Calculate kneefield of demagnetization curve. The knee field strength
        is the internal magnetic field at which the magnetization in the
        demagnetization curve is reduced to 90 % of the remanence.

        Parameters
        ----------
        None

        Returns
        -------
        kneefield: ENTITY
            Knee field strength as mammos_entity.Entity
        """
        s = self.segments()
        idx0 = np.argsort(self.M.q[s[0] : s[1]])
        Hk0 = np.interp(
            self.remanence.q * 0.9,
            self.M.q[s[0] : s[1]][idx0],
            self.H.q[s[0] : s[1]][idx0],
        )
        if len(s) >= 5:
            idx1 = np.argsort(self.M.q[s[2] : s[3]])
            Hk1 = np.interp(
                self.remanence.q * -0.9,
                self.M.q[s[2] : s[3]][idx1],
                self.H.q[s[2] : s[3]][idx1],
            )
            Hk = me.Entity("KneeField", max(np.abs(Hk) for Hk in [Hk0, Hk1]))
        else:
            Hk = me.Entity("KneeField", Hk0)
        return Hk

    def _calc_squareness(self):
        r"""
        Calculate squareness of demagnetization curve.

        .. math:: S = \frac{H_K}{H_C}

        Parameters
        ----------
        None

        Returns
        -------
        S: FLOAT
            Squareness (dimensionless)
        """
        return (self.kneefield.q / self.coercivity.q).value

    def estimate_saturation(self, threshold=None):
        """
        Calculate an estimation of the saturation magnetisation, return it and
        simultaneously assign it to the VSM object. A possible high field
        susceptibility can optionally be corrected for.

        Parameters
        ----------
        threshold: NONE | FLOAT, optional
            If a float instead of None is given, the saturation value will be
            corrected for a high field susceptibility by substracting any
            inclination at magnetisation values above the maximum field times
            the threshold factor. The saturation magnetisation is then the
            y-intersection from extrapolating M over 1/H to 0.
            Default value is None.

        Returns
        -------
        Ms: ENTITY
            Saturation magnetisation as SpontaneousMagnetization.
        """
        if threshold is None:
            Ms = []
            for segment_num in range(0, len(self.segments()), 2):
                start_idx = self.segments()[segment_num]
                if start_idx == max(self.segments()):
                    end_idx = None
                else:
                    end_idx = self.segments()[segment_num + 1]
                Ms.append(max(np.abs(self.M.q[start_idx:end_idx])))

        if threshold is not None:
            Ms = []
            for segment_num in range(0, len(self.segments()), 2):
                start_idx = self.segments()[segment_num]
                if start_idx == max(self.segments()):
                    end_idx = None
                else:
                    end_idx = self.segments()[segment_num + 1]

                H = np.abs(self.H.q[start_idx:end_idx])
                M = np.abs(self.M.q[start_idx:end_idx])
                filt = np.max(H) * threshold < H
                linreg1 = linregress(H[filt], M[filt])
                linreg2 = linregress(1 / H[filt], M[filt] - H[filt] * linreg1.slope)
                Ms.append(linreg2.intercept * mu.A / mu.m)

        Ms = me.Entity("SpontaneousMagnetization", max(Ms))
        self.saturation = Ms
        self.properties["Ms"] = Ms
        return Ms

    def segments(self, edge=0.05):
        r"""
        Find indices of segmentation points which can be used to seperate each
        measurement segment from each other. The segments are seperated by a
        changing sign of M (root) or a changing sign of dH/dt (peak).
        The segmentation points are chosen to be the index right after the root
        and right at the peak.
        Segmentation points with even indices (including 0) are peaks while
        segmentation points with uneven indices are roots.
        When plotting H over t, the cut-off points s are:
        This is an ASCII sketch and will probably not render correctly. Look
        at this method in the source code for illustration in that case.

                /|\             /|\
               / | \           / | \
              /  |  \         /  |  \
        H=0  ----|---|-------|---|---|-----> t
                 |   |\     /|   |   |\
                 |   | \   / |   |   | \
                 |   |  \|/  |   |   |
                 |   |   |   |   |   |
        s:       0   1   2   3   4   5    and so on ...

        Parameters
        ----------
        edge: FLOAT, optional
            Percentage of measurement to be treated as edge. Default is 0.05,
            which means that segmentation points closer than 1 % of the total
            measurement width to the edges will be discarded.

        Returns
        -------
         s: ARRAY[INT]
            Array of segmentation points that can be used to segmentise the
            measurement.
        """
        t = self.t.q
        h = self.H_ext.q
        m = self.M.q
        # find all roots
        r = np.nonzero(np.diff(np.sign(m)))[0] + 2
        # find peaks, we know that they must be higher than 3e6 A/m and far apart
        p = find_peaks(np.abs(h), distance=10, height=50)[0]
        # all segmentation points
        s = np.sort(np.append(r, p))
        # discard segmentation points if they're very close to start
        s = s[len(s[s < (edge * len(t))]) :]
        # discard segmentation points if they're very close to end
        if len(s[s > ((1 - edge) * len(t))]) != 0:
            s = s[: -len(s[s > ((1 - edge) * len(t))])]
        return s

    def plot(self, filepath=None, demag=True, label=None):
        """
        Plot major hysteresis loop, optionally with inset of demagnetization
        curve and save figure if a filepath is given.

        Parameters
        ----------
        filepath: STR | PATH, optional
            Filepath for saving the figure. Default is None, in that case no
            file is saved.
        demag: BOOL, optional
            Boolean that determines if demagnetization curve is plotted as an
            inset next to hysteresis loop. Default is True.
        label: STR, optional
            Optional label of hysteresis loop that can be displayed as title.
            Default is None, in that case no legend is displayed.

        Returns
        -------
        None
        """
        H = self.H.q.to("T")  # converts H from A/m to Tesla
        M = self.M.q.to("T")  # converts M from A/m to Tesla

        fig, ax1 = plt.subplots(1, 1, figsize=(16 / 2.54, 12 / 2.54))
        ax1.plot(H, M)

        # format plot
        ax1.set_xlabel(r"$\mu_0 H_{int}$ in $T$")
        ax1.set_ylabel(r"$J$ in $T$")
        if label is not None:
            ax1.set_title(label)

        # plot inset of demagnetization curve
        if demag:
            start_idx, end_idx = self.segments()[:2]
            ax2 = ax1.inset_axes([0.625, 0.15, 0.3, 0.5])
            ax2.plot(H[start_idx:end_idx], M[start_idx:end_idx])

            # format inset
            Hmin = self.coercivity.q.to("T").value * -1.1
            Jmax = self.remanence.q.to("T").value * 1.1
            ax2.axis([Hmin, 0, 0, Jmax])
            ax2.yaxis.set_label_position("right")
            ax2.set_xlabel(r"$\mu_0 H_{int}$ in $T$")
            ax2.set_ylabel(r"$J$ in $T$")

        # save figure if filepath is given
        if filepath is not None:
            fig.savefig(filepath, dpi=300)


class MH_recoil(MH):
    """
    Class for importing, storing and using of VSM-data from recoil loop M(H)
    measurements.
    Recoil loop measurements are assumed to be measurements where
    the sample is first saturated at some positive field. After this, a number
    of recoil loops are performed, where the magnetic field is reduced to some
    negative value (which often includes fields that do not fully demagnetise
    the sample) and is then increased to an external field of 0 or to a
    positive external field which increases the internal field to 0.

    Attributes
    ----------
    H: ENTITY
        Internal magnetic field as mammos_entity.Entity
    H_ext: ENTITY
        External magnetic field as mammos_entity.Entity
    M: ENTITY
        Magnetization as mammos_entity.Entity
    m: Quantity
        Magnetic Moment as mammos_units.Quantity
    T: ENTITY
        Absolute temperature as mammos_entity.Entity
    t: ENTITY
        Time as mammos_entity.Entity
    D: ENTITY
        Demagnetizing factor as mammos_entity.Entity

    Methods
    -------
    segments()
        Find indices of segmentation points which can be used to extract each
        recoil loop.
    plot()
        Plot data according to measurement type, optionally saves as png.
    to_hdf5()
        Save contents of .DAT-file and calculated properties in hdf5 file.
    """

    def segments(self, prominence=5e3):
        """
        Get indices of segmentation points. Segmentation point 0 occurs
        at H = H_max. All uneven segmentation points are the start of a recoil
        loop (only section with positive change in H) while all subsequent even
        segmentation points are the respective ends of each recoil loop.

        Parameters
        ----------
        prominence : FLOAT, optional
            The prominence which is used to find all peaks in H_ext(t). See the
            documentation of scipy.signal.find_peaks for more details.
            The default is 5e3.

        Returns
        -------
        segments: ARRAY
            Segmentation points used to extract each recoil loop.
        """
        npeaks, _ = find_peaks(-self.H_ext.q, prominence=prominence)
        # npeaks = npeaks[self.H.q[npeaks] < 0] + 1
        npeaks = npeaks + 1
        ppeaks, _ = find_peaks(self.H_ext.q, prominence=prominence)
        # ppeaks = ppeaks[self.H.q[ppeaks] < 0]
        ppeaks = np.append(ppeaks, len(self.H.q) - 1)
        apeaks = np.sort(np.append(npeaks, ppeaks))
        return apeaks

    def recoil_susceptibility(self, prominence=5e3, take_mean=True):
        """
        Calculate recoil susceptibility from recoil loop measurement as slope
        of a linear regression of each recoil loop.

        Parameters
        ----------
        prominence: FLOAT, optional
            Prominence used to find segmentation points, see MH_recoil.segments
            The default is 5e3.
        take_mean: BOOL, optional
            Wether to take the mean of all recoil susceptibilities instead of
            returning an individual one for each loop. Default is True.

        Returns
        -------
        recoil_susceptibility: FLOAT | ARRAY
            Recoil susceptibility calculated from recoil loop measurement.
        """
        segments = self.segments(prominence=prominence)
        starts = segments[1::2]
        ends = segments[2::2]
        num_loops = min([len(starts), len(ends)])

        recoil_suscep = np.array(
            [
                linregress(
                    self.H.q[starts[i] : ends[i]], self.M.q[starts[i] : ends[i]]
                ).slope
                for i in range(num_loops)
            ]
        )

        if take_mean:
            recoil_suscep = np.mean(recoil_suscep)

        return recoil_suscep

    def external_comp_field(self, prominence=5e3):
        """
        Return external compensation fields, that will lead to an endpoint
        close to an internal magnetic field of zero for each recoil loop. This
        only makes sense to be calculated from recoil loop measurements, where
        the endpoint for each recoil loop was an external magnetic field of 0,
        which will lead to the respective internal magnetic field being
        non-zero. The external compensation fields can then be used in a new
        measurement which will then enable a recoil loop measurement corrected
        for demagnetisation.\n
        External compensation field is calculated from the internal recoil
        fields (at the end of each recoil loop) and the recoil susceptibility
        according to:
            .. math:: H^e_{comp} = \\frac{- H^i_{recoil}}{1 - D * \\chi_{recoil}}

        Parameters
        ----------
        prominence: FLOAT, optional
            Prominence used to find segmentation points, see MH_recoil.segments
            The default is 5e3.

        Returns
        -------
        external_compensation_field: ARRAY
            External compensation fields, that would lead to an internal field
            close to zero at the end of each recoil loop.

        """
        # get demagnetising factor
        D = self.D.value
        # get internal recoil fields as end points of each recoil loop
        i_r_f = self.H.q[self.segments(prominence=prominence)[2::2]]
        # get recoil susceptibilities individually for each recoil loop
        r_chi = self.recoil_susceptibility(prominence=prominence, take_mean=False)

        # calculate external compensation fields
        e_c_f = -i_r_f / (1 - D * r_chi)

        return e_c_f

    def plot(self, filepath=None, label=None):
        """
        Plot M(H) measurement and save figure if a filepath is given.

        Parameters
        ----------
        filepath: STR | PATH, optional
            Filepath for saving the figure. Default is None, in that case no
            file is saved.
        label: STR, optional
            Optional label of M(H)-measurement that can be displayed as title.
            Default is None, in that case no legend is displayed.

        Returns
        -------
        None
        """
        H = self.H.q.to("T")[self.segments()[0] :]  # converts H from A/m to Tesla
        M = self.M.q.to("T")[self.segments()[0] :]  # converts M from A/m to Tesla

        fig, ax1 = plt.subplots(1, 1, figsize=(16 / 2.54, 12 / 2.54))
        ax1.plot(H, M)

        # format plot
        ax1.set_xlabel(r"$\mu_0 H_{int}$ in $T$")
        ax1.set_ylabel(r"$J$ in $T$")
        ax1.set_xlim([np.min(self.H.q.to("T")).value * 1.1, 0])
        if label is not None:
            ax1.set_title(label)

        # save figure if filepath is given
        if filepath is not None:
            fig.savefig(filepath, dpi=300)


class FORC(MH):
    """
    Class for importing, storing and using VSM data of FORC measurements.

    Attributes
    ----------
    H: ENTITY
        Internal magnetic field as mammos_entity.Entity
    H_ext: ENTITY
        External magnetic field as mammos_entity.Entity
    M: ENTITY
        Magnetization as mammos_entity.Entity
    m: Quantity
        Magnetic Moment as mammos_units.Quantity
    T: ENTITY
        Absolute temperature as mammos_entity.Entity
    t: ENTITY
        Time as mammos_entity.Entity
    D: ENTITY
        Demagnetizing factor as mammos_entity.Entity

    Methods
    -------
    to_hdf5()
        Save contents of .DAT-file and calculated properties in hdf5 file.
    segments()
        Find indices of segmentation points which can be used to extract each
        FORC from the measurement.
    forc_grid()
        Extract grid of magnetic fields, reversal fields and magnetisations
        from the FORC-measurement.
    forc_dist()
        Calculate the transformed coordinates H_c and H_i as well as the
        FORC distribution.
    """

    def __init__(self, datfile, read_method="auto"):
        super().__init__(datfile, read_method="auto")

    def segments(self, prominence=5e3):
        """
        Get indices of segmentation points. Segmentation point 0 occurs
        at H = H_max. All uneven segmentation points are the start of a FORC
        (only section with positive change in H) while all subsequent even
        segmentation points are the respective ends of each FORC.

        Parameters
        ----------
        prominence: FLOAT, optional
            The prominence which is used to find all peaks in H_ext(t). See the
            documentation of scipy.signal.find_peaks for more details.
            The default is 5e3.

        Returns
        -------
        segments: ARRAY
            Segmentation points used to extract each recoil loop.
        """
        ppeaks, _ = find_peaks(self.H_ext.q, prominence)
        ppeaks = np.append(ppeaks, len(self.H_ext.q) - 1)
        npeaks, _ = find_peaks(-self.H_ext.q + np.max(self.H_ext.q), prominence)
        npeaks = npeaks + 1
        apeaks = np.sort(np.append(ppeaks, npeaks))
        return apeaks

    def forc_grid(self, demag_correction=False, prominence=5e3):
        """
        Extract grid of magnetic fields, reversal fields and magnetisations
        from the FORC-measurement.

        Parameters
        ----------
        demag_correction: BOOL, optional
            Whether to correct the magnetic field for Demagnetisation.
            The default is True.
        prominence: FLOAT, optional
            Prominence used to find segmentation points, see FORC.segments
            The default is 5e3.

        Returns
        -------
        h_grid: QUANTITY
            Magnetic field at each grid point of the FORC-measurement.
            If demag_correction was True, this is the internal magnetic field.
            Otherwise it's the external magnetic field.
        h_r_grid: QUANTITY
            Magnetic reversal field of each FORC. Has the same shape as h_grid.
            If demag_correction was True, this is the internal magnetic field.
            Otherwise it's the external magnetic field.
        m_grid: QUANTITY
            Magnetisation at each grid point of the FORC-measurement.
        """
        if demag_correction:
            h = self.H.q
        else:
            h = self.H_ext.q
        m = self.M.q

        # get starting and end points for each FORC
        starts = self.segments(prominence=prominence)[1::2]
        ends = self.segments(prominence=prominence)[2::2]
        # find highest number of points found in any FORC, which is grid width
        grid_width = max([ends[i] - starts[i] for i in range(len(starts))])

        # make sure all FORCs have start and end point
        # total number of FORCs will be grid height
        if starts.shape == ends.shape:
            grid_height = starts.shape[0]
        else:
            raise Exception(
                "Grid height cannot be calculated. Number of"
                + " positive and negative peaks is not the same"
            )

        # initialise grids so they can be referred to in the loop
        h_grid = np.array([]) * h.unit
        h_r_grid = np.array([]) * h.unit
        m_grid = np.array([]) * m.unit

        # loop over each FORC
        for i in range(len(starts)):
            # prepare pad of nan, so that each FORC will have same (grid) width
            pad = np.array((grid_width - (ends[i] - starts[i])) * [np.nan])

            # get all H of current FORC and add pad (pad can be empty)
            current_h = np.append(pad * h.unit, h[starts[i] : ends[i]])
            # add H of current FORC to grid of all FORCs
            h_grid = np.append(h_grid, current_h)

            # get reversal field H_r of current FORC
            current_h_r = h[starts[i]]
            # H_r is the same for all points (H) of one FORC
            current_h_r = current_h_r.repeat(ends[i] - starts[i])
            # add pad to reach grid width
            current_h_r = np.append(pad * h.unit, current_h_r)
            # add H_r of current FORC to grid of all FORCs
            h_r_grid = np.append(h_r_grid, current_h_r)

            # get all M of current FORC and add pad (pad can be empty)
            current_m = np.append(pad * m.unit, m[starts[i] : ends[i]])
            # add M of current FORC to grid of all FORCs
            m_grid = np.append(m_grid, current_m)

        # reshape all flat datasets prepared in loop to actual grids
        h_grid = h_grid.reshape((grid_height, grid_width))
        h_r_grid = h_r_grid.reshape((grid_height, grid_width))
        m_grid = m_grid.reshape((grid_height, grid_width))

        return h_grid, h_r_grid, m_grid

    def forc_dist(self, SF, demag_correction=True, prominence=5e3):
        """
        Calculate the transformed coordinates H_c and H_i as well as the
        FORC distribution. Takes a grid of (2*SF+1)**2 measurement points from
        the grid of M(H_r, H) points and fits a quadratic function. The
        FORC distribution is then the mixed partial derivative of this fitted
        polynomial.

        Parameters
        ----------
        SF: INT
            Smoothing factor to be used in the calculation of the FORC
            distribution. The number of points contributing to a point in the
            FORC distribution is a square grid with a side length of (2*SF+1)
            measurement points.
        demag_correction: BOOL, optional
            Whether to correct the magnetic field for Demagnetisation.
            The default is True.
        prominence: FLOAT, optional
            Prominence used to find segmentation points, see FORC.segments
            The default is 5e3.

        Returns
        -------
        H_c: QUANTITY
            Local coercive field H_c = (H_r - H) / 2 defined for H > 0
        H_i: QUANTITY
            Local interaction field H_i = (H_r + H) / 2
        rho: ARRAY
            FORC distribution at each point of the FORC measurement grid.
        """

        # define quadratic function that will be fitted to measurement points
        def quadratic_fun(h_hr, a1, a2, a3, a4, a5, a6):
            h, hr = h_hr
            return (
                a1 + a2 * hr + a3 * hr**2 + a4 * h + a5 * h**2 + a6 * hr * h
            ).ravel()

        # get the output from FORC.forc_grid as input for this method
        h, hr, m = self.forc_grid(
            demag_correction=demag_correction, prominence=prominence
        )

        # initialise empty forc_distribution to be used in loops
        rho = np.zeros(m.shape)
        rho[:, :] = np.nan

        # loop over every FORC
        for i in range(SF, m.shape[0] - SF):
            # loop over every point in each FORC
            for j in range(SF, m.shape[1] - SF + 1):
                # extract all points that contribute to FORC-distribution
                h_grid = h[i - SF : i + SF + 1, j - SF : j + SF + 1]
                hr_grid = hr[i - SF : i + SF + 1, j - SF : j + SF + 1]
                m_grid = m[i - SF : i + SF + 1, j - SF : j + SF + 1]

                # don't calculate a FORC-distribution if any of the points in
                # the selected grid are undefined
                if np.isnan([h_grid, hr_grid, m_grid]).any():
                    continue

                # calculate FORC-distribution at selected point
                else:
                    # a6 is the last of the fitted parameters
                    a6 = curve_fit(quadratic_fun, (h_grid, hr_grid), m_grid.ravel())[0][
                        -1
                    ]
                    rho[i, j] = a6 * -0.5

        # calculate coordinate transformation from (H, H_r) to (H_c, H_i)
        H_c = np.abs((hr - h) / 2)
        H_i = (hr + h) / 2

        return H_c, H_i, rho


class MT(VSM, _Property_Container):
    """
    Class for importing, storing and using of VSM-data from M(T)-measurement
    aswell as derived properties.

    Attributes
    ----------
    H: ENTITY
        Internal magnetic field as mammos_entity.Entity
    H_ext: ENTITY
        External magnetic field as mammos_entity.Entity
    M: ENTITY
        Magnetization as mammos_entity.Entity
    m: Quantity
        Magnetic Moment as mammos_units.Quantity
    T: ENTITY
        Absolute temperature as mammos_entity.Entity
    t: ENTITY
        Time as mammos_entity.Entity
    D: ENTITY
        Demagnetizing factor as mammos_entity.Entity
    Tc: ENTITY
        Curie-temperature as mammos_entity.Entity

    Methods
    -------
    segments()
        Find indices of segmentation points which can be used to seperate each
        measurement segment from each other.
    plot()
        Plot data according to measurement type, optionally saves as png.
    properties_to_file()
        Save all properties derived from the VSM-measurement to CSV-file or to
        YAML file, using the mammos-entity io functionality.
    to_hdf5()
        Save contents of .DAT-file and calculated properties in hdf5 file.
    """

    def __init__(self, datfile, read_method="auto"):
        # import data
        super().load_qd(datfile, read_method=read_method)

        # calculate properties
        self.Tc = self._calc_Tc()
        self.properties = {"Tc": self.Tc}

    def _calc_Tc(self):
        """
        Calculate Curie-temperature from M(T) measurement, assuming only one
        Curie-temperature. Use with caution.

        Parameters
        ----------
        None

        Returns
        -------
        Tc: ENTITY
            Curie-Temperature as mammos_entity.Entity
        """
        M = self.M.q
        T = self.T.q
        # crop off measurement edges, where extreme noise usually occurs
        cM = M[(np.min(T) * 1.05 < T) * (np.max(T) * 0.95 > T)]
        cT = T[(np.min(T) * 1.05 < T) * (np.max(T) * 0.95 > T)]
        # calculate dM/dT and smooth generously
        dmdT = np.convolve(np.gradient(cM) / np.gradient(cT), np.ones(20) / 20, "same")
        # norm dM/dT to values between -1 and 1
        ndmdT = dmdT / np.max(np.abs(dmdT))
        # find peaks in normed dM/dT
        p = find_peaks(np.abs(ndmdT), distance=20, width=20, height=0.1)[0]
        # sort peaks
        Tc = np.sort(cT[p])
        # pass all peaks of normed dM/dT to Curie-Temperature
        # the user should decide themselves, whether heating or cooling is used
        Tc = me.Entity("CurieTemperature", Tc)
        return Tc

    def segments(self, edge=0.05):
        r"""
        Find indices of segmentation points which can be used to seperate each
        measurement segment from each other. The segments are seperated by a
        changing sign of dT/dt.

        Parameters
        ----------
        edge: FLOAT, optional
            Percentage of measurement to be treated as edge. Default is 0.05,
            which means that segmentation points closer than 1 % of the total
            measurement width to the edges will be discarded.

        Returns
        -------
         s: ARRAY[INT]
            Array of segmentation points that can be used to segmentise the
            measurement.
        """
        t = self.t.q
        T = self.T.q
        # find sall segmentation points (peaks of T(t))
        s = find_peaks(np.abs(T), distance=10, prominence=5)[0]
        # discard segmentation points if they're very close to start
        s = s[len(s[s < (edge * len(t))]) :]
        # discard segmentation points if they're very close to end
        if len(s[s > ((1 - edge) * len(t))]) != 0:
            s = s[: -len(s[s > ((1 - edge) * len(t))])]
        return s

    def plot(self, filepath=None):
        """
        Plot cooling curve of M(T) measurement. Save to file if path is given.

        Parameters
        ----------
        filepath : STR | PATH, optional
            Filepath for saving the figure. Default is None, in that case no
            file is saved.

        Returns
        -------
        None.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.plot(self.T.q, self.M.q)
        ax1.set_ylabel(f"Magnetization in {self.M.unit}")

        M = self.M.q
        T = self.T.q
        # crop off measurement edges, where extreme noise usually occurs
        cM = M[(np.min(T) * 1.05 < T) * (np.max(T) * 0.95 > T)]
        cT = T[(np.min(T) * 1.05 < T) * (np.max(T) * 0.95 > T)]
        # calculate dM/dT and smooth generously
        dmdT = np.convolve(np.gradient(cM) / np.gradient(cT), np.ones(20) / 20, "same")
        ax2.plot(cT, dmdT)
        ax2.set_ylabel("$\\frac{dM}{dT}$")
        ax2.set_xlabel(f"Temperature in {self.T.unit}")

        if filepath is not None:
            fig.savefig(filepath, dpi=300)


def plot_multiple_MH_major(data, filepath=None, labels=None, demag=True):
    """
    Plot hysteresis loops and optionally demagnetization curves of
    several major hysteresis loop measurements.
    Saves figure if filepath is given.

    Parameters
    ----------
    data: LIST[MH_major]
        List of several objects which have to be of the MH_major class.
    filepath: STR | PATH, optional
        Filepath for saving the figure. Default is None, in that case no file
        is saved.
    labels: LIST, optional
        List of labels that are going to be used in the legend of the plot.
        Has to have the same length as data. If none is given then the names
        of the .DAT files each VSM object was calculated from will be used.
    demag: BOOL, optional
        Boolean that determines if demagnetization curve is plotted as an
        inset next to hysteresis loop. Default is True.

    Returns
    -------
    None
    """
    if labels is None:
        labels = [vsm.path.stem for vsm in data]

    fig, ax1 = plt.subplots()

    # plot hysteresis loops
    for i in range(len(data)):
        ax1.plot(data[i].H.q.to("T"), data[i].M.q.to("T"), label=labels[i])

    # format plot
    ax1.set_xlabel(r"$\mu_0 H_{int}$ in $T$")
    ax1.set_ylabel(r"$J$ in $T$")
    ax1.legend()

    # plot inset of demagnetization curves
    if demag:
        ax2 = ax1.inset_axes([0.58, 0.15, 0.3, 0.5])
        for i in range(len(data)):
            start_idx, end_idx = data[i].segments()[:2]
            ax2.plot(
                data[i].H.q.to("T")[start_idx:end_idx],
                data[i].M.q.to("T")[start_idx:end_idx],
                label=labels[i],
            )

        # format plot
        Hmin = max([i.coercivity.q.to("T") for i in data]).value * -1.1
        Jmax = max([i.remanence.q.to("T") for i in data]).value * 1.1
        ax2.axis([Hmin, 0, 0, Jmax])
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.set_xlabel(r"$\mu_0 H_{int}$ in $T$")
        ax2.set_ylabel(r"$J$ in $T$")

    # save figure if filepath is given
    if filepath is not None:
        fig.savefig(filepath, dpi=300)
    return None


def mult_properties_to_file(data, filepath, labels=None):
    """
    Save all properties derived from list of VSM-measurements to CSV-file
    or to YAML file, using the mammos-entity io functionality.

    Parameters
    ----------
    data: LIST[MH_major | MT]
        List of VSM objects that will have their properties exported.
    filepath: STR | PATH
        Filepath to save the file to. Ending also determines type of file.
    labels: LIST[STR], optional
        List of strings to identify the VSM objects more easily.
        If not given, the filenames of each VSM-object are used instead.

    Returns
    -------
    None
    """
    # check if all objects in data are of the same type
    ref_type = type(data[0])
    if not all([type(vsm) is ref_type for vsm in data]):
        raise Exception(
            "Objects in data are not all of the same type.\n"
            + "Please only export the properties from objects of\n"
            + "the same type together."
        )

    description = (
        f"mammos-entity version = {version('mammos-entity')}\n"
        + f"magmeas       version = {version('magmeas')}"
    )

    if labels is None:
        labels = [vsm.path.stem for vsm in data]

    if all([isinstance(vsm, MH_major) for vsm in data]) and all(
        [hasattr(vsm, "saturation") for vsm in data]
    ):
        me.io.entities_to_file(
            filepath,
            description,
            labels=labels,
            Ms=me.concat_flat([vsm.saturation for vsm in data]),
            Mr=me.concat_flat([vsm.remanence for vsm in data]),
            Hc=me.concat_flat([vsm.coercivity for vsm in data]),
            BHmax=me.concat_flat([vsm.BHmax for vsm in data]),
            Hk=me.concat_flat([vsm.kneefield for vsm in data]),
        )

    elif all([isinstance(vsm, MH_major) for vsm in data]) and any(
        [not hasattr(vsm, "saturation") for vsm in data]
    ):
        me.io.entities_to_file(
            filepath,
            description,
            labels=labels,
            Mr=me.concat_flat([vsm.remanence for vsm in data]),
            Hc=me.concat_flat([vsm.coercivity for vsm in data]),
            BHmax=me.concat_flat([vsm.BHmax for vsm in data]),
            Hk=me.concat_flat([vsm.kneefield for vsm in data]),
        )

    elif all([isinstance(vsm, MT) for vsm in data]):
        me.io.entities_to_file(
            filepath,
            description,
            labels=labels,
            Tc=me.concat_flat([vsm.Tc for vsm in data]),
        )
