"""VSM class and functions using it."""

from pathlib import Path

import h5py
import mammos_entity as me
import mammos_units as mu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from magmeas.utils import droot, rextract

mu_0 = mu.constants.mu0
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
    plot()
        Plot data according to measurement type, optionally saves as png.
    properties_to_txt()
        Saves all properties derived from VSM-measurement to CSV-file.
    properties_to_json()
        Saves all properties derived from VSM-measurement to JSON-file.
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
            self.remanence = self._calc_remanence()
            self.coercivity = self._calc_coercivity()
            self.BHmax = self._calc_BHmax()
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
            with open(datfile, "rb") as f:
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
        df = pd.read_csv(datfile, skiprows=34, encoding="cp1252")
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

    def _calc_remanence(self):
        """
        Extract remanent magnetization from hysteresis loop.

        Parameters
        ----------
        NONE

        Returns
        -------
        remanence: ENTITY
            Remanent magnetization as mammos_entity.Entity object
        """
        # find intersections of hysteresis loop with H=0
        a = droot(self.M.q, self.H.q)
        a = np.abs(a)  # get absolute values of all intersections

        # test for initial magnetization curve, in this case the interception
        # point of the hysteresis will be lower than the remanence and thus
        # discarded
        # a deviation of 2 % has been arbitrarily defined to distinguish the
        # interception during the initial magnetization from the remanences
        if np.abs((a[0] - np.mean(a[1:])) / np.mean(a[1:])) > 0.02:
            a = a[1:]
        # average all interception points at H=0 to one mean remanence
        a = np.mean(a)
        # save the remanence as Entity
        Mr = me.Entity("Remanence", a)
        return Mr

    def _calc_coercivity(self):
        """
        Extract internal coercivity from hysteresis loop.

        Parameters
        ----------
        NONE

        Returns
        -------
        coercivity: ENTITY
            Internal coercivity as mammos_entity.Entity object
        """
        # find intersections of hysteresis loop with M=0
        a = droot(self.H.q, self.M.q)
        a = np.abs(a)  # get absolute values of all coercivities
        # test for initial magnetization curve, in this case the interception
        # point of the hysteresis will deviate from coercivity, thus discarded
        # a deviation of 2 % has been arbitrarily defined to distinguish the
        # interception during the initial magnetization from the coercivity
        if np.abs((a[0] - np.mean(a[1:])) / np.mean(a[1:])) > 0.02:
            a = a[1:]
        # average all interception points at M=0 to one mean coercivity
        a = np.mean(a)
        # save the coercivity as Entity
        iHc = me.Entity("CoercivityHc", a)
        return iHc

    def _calc_BHmax(self):
        """
        Extract maximum Energy product from demagnetization curve.

        Parameters
        ----------
        None

        Returns
        -------
        BHmax: ENTITY
            Maximum energy product as mammos_entity.Entity object
        """
        # calculate BH
        BH = (self.H.q + self.M.q) * mu_0 * self.H.q
        # product of B and H is positive in first and third quadrant, negative
        # in second and fourth quadrant, so no finding of demagnetization curve
        # is necessary, it will always be found at negative values
        # BHmax is minimum of BH
        # BH at BHmax should always be negative value
        # As is common practice, we return and save BHmax with a positive value
        BHmax = me.Entity("MaximumEnergyProduct", np.min(BH) * -1)
        return BHmax

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
        # value that magnetization is supposed to have at knee-point
        Mk = 0.9 * self.remanence.q
        # find intersections of Hysteresis loop with M=Mk
        a = droot(self.H.q, self.M.q - Mk)
        a = np.abs(a)  # get absolute values
        a = a[1]  # second root should be knee field strength
        Hk = me.Entity("KneeField", a)
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
        # norm the magnetization to positive values between 0 and 1
        # we are only interested in Tc, absolute moments don't matter
        nM = np.abs(self.M.q) / np.max(np.abs(self.M.q))

        # let's only look at M(T) during cooling, this is usually more reliable
        # cooling is assumed to occur after half of the measurement time
        # also cut off last couple of measurement points as they are unstable
        selec = (self.t.q > np.max(self.t.q) * 0.5) * (
            self.t.q < np.max(self.t.q) * 0.9
        )
        nM = nM[selec]
        T = self.T.q[selec]
        # generous kernel for smoothing
        kernel = np.ones(20) / 20
        # smooth measurement by convolution with kernel
        sT = np.convolve(T, kernel, mode="valid")
        sM = np.convolve(nM, kernel, mode="valid")
        # Tc is temperature where dM/dT has minimum
        Tc = sT[np.argmin(np.gradient(sM) / np.gradient(sT))]
        Tc = me.Entity("CurieTemperature", Tc)
        return Tc

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
            ax2 = ax1.inset_axes([0.625, 0.15, 0.3, 0.5])
            ax2.plot(H[100:-100], M[100:-100], label=label)

            # format inset
            Hmin = self.coercivity.q.to("T").value * -1.1
            Jmax = self.remanence.q.to("T").value * 1.1
            ax2.axis([Hmin, 0, 0, Jmax])
            ax2.yaxis.set_label_position("right")
            ax2.set_xlabel(r"$\mu_0 H_{int}$ in $T$")
            ax2.set_ylabel(r"$J$ in $T$")

        # save figure if filepath is given
        if filepath is not None:
            plt.savefig(filepath, dpi=300)

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
        fig, ax = plt.subplots()

        ax.plot(
            self.T.q[self.t.q > max(self.t.q) / 2],
            self.M.q[self.t.q > max(self.t.q) / 2],
        )

        ax.set_xlabel(f"Temperature in {self.T.unit}")
        ax.set_ylabel(f"Magnetization in {self.M.unit}")
        ax.xaxis.set_inverted(True)

        if filepath is not None:
            plt.savefig(filepath, dpi=300)

    def properties_to_file(self, filepath):
        r"""
        Save all properties derived from the VSM-measurement to CSV-file.

        Parameters
        ----------
        filepath: STR | PATH
            Fielpath to save the TXT-file to.

        Returns
        -------
        None
        """
        if self.measurement == "M(H)":
            me.io.entities_to_file(
                filepath,
                Mr=self.remanence,
                Hc=self.coercivity,
                BHmax=self.BHmax,
                Hk=self.kneefield,
            )
        elif self.measurement == "M(T)":
            me.io.entities_to_file(filepath, Tc=self.Tc)

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

    def to_hdf5(self, unit="T"):
        """
        Save .DAT-file and calculated properties in hdf5 file.

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

        f = h5py.File(self.path.parent.joinpath(self.path.stem + ".hdf5"), "a")
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
            f.create_dataset(
                "Properties/Remanence", data=self.remanence.q.to(unit).value
            )
            f["Properties/Remanence"].attrs["unit"] = unit
            f.create_dataset(
                "Properties/Coercivity", data=self.coercivity.q.to(unit).value
            )
            f["Properties/Coercivity"].attrs["unit"] = unit
            f.create_dataset("Properties/BHmax", data=self.BHmax.q.to(mu.kJ / mu.m**3))
            f["Properties/BHmax"].attrs["unit"] = "kJ/m^3"
            f.create_dataset("Properties/Squareness", data=self.squareness())
        elif self.measurement == "M(T)":
            f.create_dataset("Properties/Tc", data=self.Tc.q.value)
            f["Properties/Tc"].attrs["unit"] = "K"


def plot_multiple_VSM(data, labels, filepath=None, demag=True):
    """
    Plot hysteresis loops and optionally demagnetization curves of
    several VSM measurements. Saves figure if filepath is given.

    Parameters
    ----------
    data: LIST
        List of several objects which have to be of the VSM class.
    labels: LIST
        List of labels that are going to be used in the legend of the plot.
        Has to have the same length as data.
    filepath: STR | PATH, optional
        Filepath for saving the figure. Default is None, in that case no file
        is saved.
    demag: BOOL, optional
        Boolean that determines if demagnetization curve is plotted as an
        inset next to hysteresis loop. Default is True.

    Returns
    -------
    None
    """
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
            ax2.plot(
                data[i].H.q.to("T")[100:-100],
                data[i].M.q.to("T")[100:-100],
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
        plt.savefig(filepath, dpi=300)
    return None
