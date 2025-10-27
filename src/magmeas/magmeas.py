"""VSM class and functions using it."""

from pathlib import Path

import h5py
import mammos_entity as me
import mammos_units as mu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mammos_analysis.hysteresis import extrinsic_properties
from scipy.signal import find_peaks

from magmeas.utils import rextract

mu.set_enabled_equivalencies(mu.magnetic_flux_field())


class VSM:
    """
    Class for importing, storing and using of VSM-data aswell as derived
    parameters.

    Attributes
    ----------
    H: ENTITY
        Internal magnetic field as mammos_entity.Entity
    M: ENTITY
        Magnetization as mammos_entity.Entity
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
    Tc: ENTITY
        Curie-temperature as mammos_entity.Entity

    Methods
    -------
    load_qd()
        Load VSM-data from a quantum design .DAT file
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

    def __init__(self, datfile, read_method="auto", calc_properties=True):
        # import data
        self.load_qd(datfile, read_method=read_method)

        # Determine type of measurement
        if self._H_var and not self._T_var:
            self.measurement = "M(H)"
        elif self._T_var:
            self.measurement = "M(T)"
        else:
            self.measurement = "unknown"

        # calculate properties
        if calc_properties and self.measurement == "M(H)":
            s = self.segments()
            idx0 = np.argsort(self.H.q[s[0] : s[2]])
            prop0 = extrinsic_properties(
                self.H.q[s[0] : s[2]][idx0], self.M.q[s[0] : s[2]][idx0], 0
            )
            if len(s) >= 5:
                idx1 = np.argsort(self.H.q[s[2] : s[4]])
                prop1 = extrinsic_properties(
                    self.H.q[s[2] : s[4]][idx1], self.M.q[s[2] : s[4]][idx1], 0
                )
                self.remanence = me.Entity("Remanence", max([prop0.Mr.q, prop1.Mr.q]))
                self.coercivity = me.Entity(
                    "CoercivityHc", max([prop0.Hc.q, prop1.Hc.q])
                )
                self.BHmax = me.Entity(
                    "MaximumEnergyProduct",
                    max([prop0.BHmax.q, prop1.BHmax.q]),
                    "kJ / m3",
                )
            else:
                self.remanence = prop0.Mr
                self.coercivity = prop0.Hc
                self.BHmax = me.Entity("MaximumEnergyProduct", prop0.BHmax, "kJ / m3")

            self.kneefield = self._calc_kneefield()
            self.squareness = self._calc_squareness()
        elif calc_properties and self.measurement == "M(T)":
            self.Tc = self._calc_Tc()

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
        # Automatically read out sample parameters
        if read_method == "auto":
            with open(self.path, "rb") as f:
                s = str(f.read(-1))
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
        df = pd.read_csv(self.path, skiprows=34, encoding="cp1252")
        # extract magnetic moment
        # So far this is only a Quantity and no Entity because the wrong unit
        # seems to be defined for the me.Entity('MagneticMoment')
        m = np.array(df["Moment (emu)"]) * mu.erg / mu.G
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
        nanfilter = ~np.isnan(H.q) * ~np.isnan(M.q) * ~np.isnan(T.q) * ~np.isnan(t.q)
        # delete all datapoints where H, M, T or t are nan and assign them to object
        self.H = me.Entity(H.ontology_label, H.q.to("A/m")[nanfilter])
        self.M = me.Entity(M.ontology_label, M.q.to("A/m")[nanfilter])
        self.T = me.Entity(T.ontology_label, T.q[nanfilter])
        # convert time stamp to time since measurement start
        self.t = me.Entity(t.ontology_label, t.q[nanfilter] - t.q[nanfilter][0])

        # Does H vary by more than 10 A/m?
        self._H_var = (np.max(self.H.q) - np.min(self.H.q)) > 10 * mu.A / mu.m
        # Does T vary by more than 10 K?
        self._T_var = (np.max(self.T.q) - np.min(self.T.q)) > 10 * mu.K

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
        return f"{self.__module__.split('.')[0]}.VSM({self.path.name})"

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
        kernel_size: INT, optional
            Width of smoothing kernel to be used to make sure, that noise in
            measurement does not accidentally introduce more segments than
            needed. The default is 10. Set to None for no smoothing.
        edge: FLOAT, optional
            Percentage of measurement to be treated as edge. Default is 0.01,
            which means that segmentation points closer than 1 % of the total
            measurement width to the edges will be discarded.

        Returns
        -------
         s: ARRAY[INT]
            Array of segmentation points that can be used to segmentise the
            measurement.
        """
        t = self.t.q
        h = self.H.q
        m = self.M.q
        # find all roots
        r = np.nonzero(np.diff(np.sign(m)))[0] + 2
        # find peaks, we know that they must be higher than 3e6 A/m and far apart
        p = find_peaks(np.abs(h), distance=100, height=3e6)[0]
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
        Plot M(H) or M(T) measurement. Wrapper function.

        Parameters
        ----------
        filepath : STR | PATH, optional
            Filepath for saving the figure. Default is None, in that case no
            file is saved.
        demag : BOOL, optional
            Boolean that determines if demagnetization curve is plotted as an
            inset next to hysteresis loop. Default is True. Only applies to
            M(H)-measurements.
        label : STR, optional
            Optional label of hysteresis loop that can be displayed in the
            legend. Default is None, in that case no legend is displayed.

        Returns
        -------
        None.
        """
        if self.measurement == "M(H)":
            self._plot_MH(filepath=filepath, demag=demag, label=label)
        elif self.measurement == "M(T)":
            self._plot_MT(filepath=filepath)
        plt.show()

    def _plot_MH(self, filepath=None, demag=True, label=None):
        """
        Plot hysteresis loop, optionally with inset of demagnetization curve
        and save figure if a filepath is given.

        Parameters
        ----------
        filepath: STR | PATH, optional
            Filepath for saving the figure. Default is None, in that case no
            file is saved.
        demag: BOOL, optional
            Boolean that determines if demagnetization curve is plotted as an
            inset next to hysteresis loop. Default is True.
        label: STR, optional
            Optional label of hysteresis loop that can be displayed in the
            legend. Default is None, in that case no legend is displayed.

        Returns
        -------
        None
        """
        H = self.H.q.to("T")  # converts H from A/m to Tesla
        M = self.M.q.to("T")  # converts M from A/m to Tesla

        fig, ax1 = plt.subplots(1, 1, figsize=(16 / 2.54, 12 / 2.54))
        ax1.plot(H, M, label=label)

        # format plot
        ax1.set_xlabel(r"$\mu_0 H_{int}$ in $T$")
        ax1.set_ylabel(r"$J$ in $T$")
        if label is not None:
            ax1.legend()

        # plot inset of demagnetization curve
        if demag:
            start_idx, end_idx = self.segments()[:2]
            ax2 = ax1.inset_axes([0.625, 0.15, 0.3, 0.5])
            ax2.plot(H[start_idx:end_idx], M[start_idx:end_idx], label=label)

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

    def _plot_MT(self, filepath=None):
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

    def properties_to_file(self, filepath, label=None):
        r"""
        Save all properties derived from the VSM-measurement to CSV-file or to
        YAML file, using the mammos-entity io functionality.

        Parameters
        ----------
        filepath: STR | PATH
            Filepyth to save the file to. Ending also determines type of file.
        label: STR, optional
            Label to be used in the description of file to be exported to.
            If none is given then the name of the .DAT file the VSM object was
            calculated from is used.

        Returns
        -------
        None
        """
        if label is None:
            description = self.path.stem
        if label is not None:
            description = label
        if self.measurement == "M(H)":
            me.io.entities_to_file(
                filepath,
                description,
                Mr=self.remanence,
                Hc=self.coercivity,
                BHmax=self.BHmax,
                Hk=self.kneefield,
            )
        elif self.measurement == "M(T)":
            me.io.entities_to_file(filepath, description, Tc=self.Tc)

    def print_properties(self, unit="T"):
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

        if self.measurement == "M(H)":
            properties = {
                "Remanence": [self.remanence.q.to(unit)],
                "Coercivity": [self.coercivity.q.to(unit)],
                "BHmax": [self.BHmax.q.to(mu.kJ / mu.m**3)],
                "S": [self.squareness],
            }
        elif self.measurement == "M(T)":
            properties = {"Tc": [self.Tc.q.to("K")]}
        for key in properties:
            print(f"{key} = {properties[key][0]}")

    def to_hdf5(self):
        """
        Save contents of .DAT-file and calculated properties in hdf5 file.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        with open(self.path) as f:
            s = str(f.read(-1))
        head = s[s.index("INFO") : s.rindex("\nDATATYPE")]
        head = head.split("\n")
        info = {
            i[i.rindex(",") + 1 :]: i[i.index(",") + 1 : i.rindex(",")] for i in head
        }
        df = pd.read_csv(self.path, skiprows=34, encoding="cp1252")

        with h5py.File(self.path.parent.joinpath(self.path.stem + ".hdf5"), "a") as f:
            for i in info:
                f.create_dataset("Info/" + i, data=info[i])
            f["Info/SAMPLE_MASS"].attrs["unit"] = "mg"
            f["Info/SAMPLE_SIZE"].attrs["unit"] = "mm"

            for i in df.columns:
                dat = np.array(df[i])
                if dat.dtype != "O":
                    f.create_dataset("Data/" + i, data=dat)
                if dat.dtype == "O":
                    f.create_dataset("Data/" + i, data=[str(j) for j in df[i]])

            if self.measurement == "M(H)":
                f.create_dataset("Properties/Remanence", data=self.remanence.value)
                f["Properties/Remanence"].attrs["unit"] = str(self.remanence.unit)
                f["Properties/Remanence"].attrs["IRI"] = self.remanence.ontology.iri
                f.create_dataset("Properties/Coercivity", data=self.coercivity.value)
                f["Properties/Coercivity"].attrs["unit"] = str(self.coercivity.unit)
                f["Properties/Coercivity"].attrs["IRI"] = self.coercivity.ontology.iri
                f.create_dataset("Properties/BHmax", data=self.BHmax.value)
                f["Properties/BHmax"].attrs["unit"] = str(self.BHmax.unit)
                f["Properties/BHmax"].attrs["IRI"] = self.BHmax.ontology.iri
                f.create_dataset("Properties/KneeField", data=self.kneefield.value)
                f["Properties/KneeField"].attrs["unit"] = str(self.kneefield.unit)
                f["Properties/KneeField"].attrs["IRI"] = self.kneefield.ontology.iri
                f.create_dataset("Properties/Squareness", data=self.squareness)
            elif self.measurement == "M(T)":
                f.create_dataset("Properties/Tc", data=self.Tc.value)
                f["Properties/Tc"].attrs["unit"] = str(self.Tc.unit)
                f["Properties/Tc"].attrs["IRI"] = self.Tc.ontology.iri


def plot_multiple_VSM(data, filepath=None, labels=None, demag=True):
    """
    Plot hysteresis loops and optionally demagnetization curves of
    several VSM measurements. Saves figure if filepath is given.

    Parameters
    ----------
    data: LIST
        List of several objects which have to be of the VSM class.
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
    plt.gca().set_prop_cycle(None)

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
    data: LIST[VSM]
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
    file_ext = filepath.name.split(".")[1].lower()
    if labels is not None and any(file_ext == e for e in ["yaml", "yml"]):
        description = labels
    if labels is None and any(file_ext == e for e in ["yaml", "yml"]):
        description = [vsm.path.stem for vsm in data]
    if labels is not None and file_ext == "csv":
        description = ""
        for label in labels:
            description = description + label + "\n"
    if labels is None and file_ext == "csv":
        description = ""
        for label in [vsm.path.stem for vsm in data]:
            description = description + label + "\n"

    if all([vsm.measurement == "M(H)" for vsm in data]):
        me.io.entities_to_file(
            filepath,
            description,
            Mr=me.Entity("Remanence", [vsm.remanence.q for vsm in data]),
            Hc=me.Entity("CoercivityHc", [vsm.coercivity.q for vsm in data]),
            BHmax=me.Entity("MaximumEnergyProduct", [vsm.BHmax.q for vsm in data]),
            Hk=me.Entity("KneeField", [vsm.kneefield.q for vsm in data]),
        )
    elif all([vsm.measurement == "M(T)" for vsm in data]):
        me.io.entities_to_file(
            filepath,
            description,
            Tc=me.Entity("CurieTemperature", [vsm.Tc.q for vsm in data]),
        )
    else:
        raise Exception(
            "VSM objects are incompatible. Please make sure\
                         not to mix M(H) and M(T) measurements during export."
        )
