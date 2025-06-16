"""VSM class and functions using it."""

import json
from pathlib import Path

import h5py
import mammos_entity as me
import mammos_units as mu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

mu_0 = mu.constants.mu0
mu.set_enabled_equivalencies(mu.magnetic_flux_field())


class VSM:
    """
    Class for importing, storing and using of VSM-data aswell as derived
    parameters.

    Attributes
    ----------
    H: Quantity
        Internal magnetic field as mammos_units.Quantity
    M: QUANTITY
        Magnetization as mammos_units.Quantity
    T: QUANTITY
        Absolute temperature as mammos_units.Quantity
    t: QUANTITY
        Time as mammos_units.Quantity
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
        Plots hysteresis loop, optionally with inset of demagnetisation curve,
        optionally saves as png
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

        if calc_properties:
            # calculate properties
            if self.measurement == "M(H)":
                self.remanence = self._calc_remanence()
                self.coercivity = self._calc_coercivity()
                self.BHmax = self._calc_BHmax()
                self.kneefield = self._calc_kneefield()
                self.squareness = self._calc_squareness()
            elif self.measurement == "M(T)":
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
        return me.Entity("DemagnetizingFactor", D)

    def load_qd(self, datfile, read_method):
        """
        Load VSM-data from a quantum systems .DAT file.

        Parameters
        ----------
        datfile: STRING
            Path to quantum systems .DAT file that data is supposed to be
            imported from

        Returns
        -------
        None

        """
        self.path = Path(datfile)

        def rextract(string, startsub, endsub):
            endind = string.index(endsub)
            startind = string.rindex(startsub, 0, endind)
            return string[startind + len(startsub) : endind]

        err = """
              Sample parameters could not be read automatically.
              Please enter sample parameters manually or enter them
              correctly in the .DAT file like this:
              INFO,<mass in mg>,SAMPLE_MASS
              INFO,(<a>, <b>, <c>),SAMPLE_SIZE
              sample dimensions a, b and c in mm, c parallel to field
              """

        if read_method == "auto":
            # Automatically read out sample parameters
            with open(datfile, "rb") as f:
                s = str(f.read(-1))

            try:
                mass = float(rextract(s, "INFO,", ",SAMPLE_MASS")) * mu.mg
            except ValueError:
                raise Exception(err) from None

            try:
                dim = rextract(s, "INFO,(", "),SAMPLE_SIZE").split(",")
                dim = np.array([float(f) for f in dim]) * mu.mm
            except ValueError:
                raise Exception(err) from None

        # Input sample parameters manually
        elif read_method == "manual":
            print("Manual input method selected")
            mass = float(input("Sample mass in mg: ")) * mu.mg
            print(f"mass = {mass}")
            a = float(input("Sample dimension a (perpendicular to field) in mm: "))
            b = float(input("Sample dimension b (perpendicular to field) in mm: "))
            c = float(input("Sample dimension c (parallel to field) in mm: "))
            dim = np.array([a, b, c]) * mu.mm
        else:
            raise Exception(err)

        density = mass / (np.prod(dim.value) * dim.unit**3)  # calculate density
        D = self._demag_prism(dim)
        # import measurement data
        df = pd.read_csv(datfile, skiprows=34, encoding="cp1252")
        # save magnetic moment
        m = np.array(df["Moment (emu)"]) * mu.erg / mu.G
        # save external magnetic field
        eH = np.array(df["Magnetic Field (Oe)"]) * mu.Oe
        # convert from Oe to A/m
        eH = eH.to("A/m")
        # calculate magnetisation
        M = m / mass * density
        # convert magnetization to reasonable units
        M = M.to("A/m")
        # calculate internal magnetic field
        H = eH - D * M
        # save absolute temperature
        T = np.array(df["Temperature (K)"]) * mu.K
        # save time stamp
        t = np.array(df["Time Stamp (sec)"]) * mu.s

        # test datapoints for missing values (where value is nan)
        nanfilter = ~np.isnan(H) * ~np.isnan(M) * ~np.isnan(T) * ~np.isnan(t)
        # delete all datapoints where any of H, M or T ar nan and assign them
        self.H = H[nanfilter]
        self.M = M[nanfilter]
        self.T = T[nanfilter]
        # convert time stamp to time since measurement start
        self.t = t[nanfilter] - t[nanfilter][0]

        # Does H vary by more than 10 A/m?
        self._H_var = (np.max(self.H) - np.min(self.H)) > 10 * mu.A / mu.m
        # Does T vary by more than 10 K?
        self._T_var = (np.max(self.T) - np.min(self.T)) > 10 * mu.K

    def _calc_remanence(self):
        """
        Extract remanent magnetization or polarization from hysteresis loop.

        Parameters
        ----------
        NONE

        Returns
        -------
        remanence: DICTIONARY
            Dictionary of possible units as keys and the respective value of
            the remanence in this unit
        """
        # find intersections of hysteresis loop with H=0
        a = droot(self.M, self.H)
        a = np.abs(a)  # get absolute values of all remanences

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
        Extract intrinsic Coercivity from hysteresis loop.

        Parameters
        ----------
        NONE

        Returns
        -------
        coercivity: DICTIONARY
            Dictionary of possible units as keys and the respective value of
            the coercivity in this unit
        """
        # find intersections of hysteresis loop with M=0
        a = droot(self.H, self.M)
        a = np.abs(a)  # get absolute values of all coercivities
        # test for initial magnetization curve, in this case the interception
        # point of the hysteresis will deviate from coercivity, thus discarded
        # a deviation of 2 % has been arbitrarily defined to distinguish the
        # interception during the initial magnetization from the coercivity
        if np.abs((a[0] - np.mean(a[1:])) / np.mean(a[1:])) > 0.02:
            a = a[1:]
        # average all interception points at M=0 to one mean coercivity
        a = np.mean(a)
        # save the remanence in different units to be called directly without
        # need for later conversion
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
        BHmax: FLOAT
            Maximum energy product in kJ/m^3
        """
        BH = (self.H + self.M) * mu_0 * self.H  # calculate BH
        # product of B and H is positive in first and third quadrant, negative
        # in second and third quadrant, so no finding of demagnetization curve
        # is necessary
        # BHmax is minimum of BH, should always be negative value
        BHmax = me.Entity("MaximumEnergyProduct", np.min(BH) * -1)
        return BHmax

    def _calc_kneefield(self):
        """
        Calculate kneefield of demagnetization curve.

        Parameters
        ----------
        None

        Returns
        -------
        kneefield: ENTITY
            knee field strength as mammos_entity.Entity
        """
        # value that magnetization is supposed to have at knee-point
        Mk = 0.9 * self.remanence
        # find intersections of Hysteresis loop with M=Mk
        a = droot(self.H, self.M - Mk)
        a = np.abs(a)  # get absolute values
        a = a[1]  # second root should be knee field strength
        Hk = me.Entity("KneeField", a)
        return Hk

    def _calc_squareness(self):
        """
        Calculate squareness of demagnetization curve.

        Parameters
        ----------
        None

        Returns
        -------
        S: FLOAT
        Squareness (dimensionless)
        """
        return self.kneefield / self.coercivity

    def _calc_Tc(self):
        """
        Calculate Curie-temperature from M(T) measurement, assuming only one
        Curie-temperature.

        Parameters
        ----------
        None

        Returns
        -------
        Tc: FLOAT
            Curie-Temperature in K
        """
        # norm the moment to positive values between 0 and 1
        # we are only interested in Tc, absolute moments don't matter
        nM = np.abs(self.M) / np.max(np.abs(self.M))

        # let's only look at M(T) during cooling, this is usually more reliable
        # cooling is assumed to occur after half of the measurement time
        # also cut off last couple of measurement points as they are unstable
        selec = (self.t > np.max(self.t) * 0.5) * (self.t < np.max(self.t) * 0.9)
        nM = nM[selec]
        T = self.T[selec]
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
        filepath : TYPE, optional
            Filepath for saving the figure. Default is None, in that case no
            file is saved.
        demag : TYPE, optional
            Boolean that determines if demagnetization curve is plotted as an
            inset next to hysteresis loop. Default is True. Only applies to
            M(H)-measurements.
        label : TYPE, optional
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
        filepath: STRING, optional
            Filepath for saving the figure. Default is None, in that case no
            file is saved.
        demag: BOOL, optional
            Boolean that determines if demagnetization curve is plotted as an
            inset next to hysteresis loop. Default is True.
        label: STRING, optional
            Optional label of hysteresis loop that can be displayed in the
            legend. Default is None, in that case no legend is displayed.

        Returns
        -------
        None
        """
        H = self.H.to("T")  # converts H from A/m to Tesla
        M = self.M.to("T")  # converts M from A/m to Tesla

        fig, ax1 = plt.subplots(1, 1, figsize=(16 / 2.54, 12 / 2.54))
        ax1.plot(H, M, label=label)

        # format plot
        ax1.xaxis.set_major_locator(MultipleLocator(2))
        ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax1.yaxis.set_major_locator(MultipleLocator(0.2))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax1.set_xlabel(r"$\mu_0 H_{int}$ in $T$")
        ax1.set_ylabel(r"$J$ in $T$")
        ax1.grid(visible=True, which="major", axis="both", linestyle=":", linewidth=1)
        ax1.grid(visible=True, which="minor", axis="both", linestyle=":", linewidth=0.5)
        if label is not None:
            ax1.legend()
        fig.tight_layout()

        # plot inset of demagnetization curve
        if demag:
            ax2 = ax1.inset_axes([0.58, 0.15, 0.3, 0.5])
            ax2.plot(H[100:-100], M[100:-100], label=label)

            # find upper and lower border of plot, so that demagnetization
            # curve fits nicely
            Hmin, Jmax = None, None
            for i in np.arange(0, 5, 0.05):
                if Hmin is None and -i <= -self.get_coercivity("T") - 0.02:
                    Hmin = -i
                if Jmax is None and i >= self.get_remanence("T") + 0.02:
                    Jmax = i
                if Hmin is not None and Jmax is not None:
                    break

            # format plot
            ax2.axis([Hmin, 0, 0, Jmax])
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()
            ax2.xaxis.set_major_locator(MultipleLocator(0.1))
            ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
            ax2.yaxis.set_major_locator(MultipleLocator(0.1))
            ax2.yaxis.set_minor_locator(MultipleLocator(0.05))
            ax2.set_xlabel(r"$\mu_0 H_{int}$ in $T$")
            ax2.set_ylabel(r"$J$ in $T$")
            ax2.grid(
                visible=True, which="major", axis="both", linestyle=":", linewidth=1
            )
            ax2.grid(
                visible=True, which="minor", axis="both", linestyle=":", linewidth=0.5
            )
            fig.tight_layout()

        # save figure if filepath is given
        if filepath is not None:
            plt.savefig(filepath, dpi=300)

    def _plot_MT(self, filepath=None):
        """
        Plot cooling curve of M(T) measurement. Save to file if path is given.

        Parameters
        ----------
        filepath : TYPE, optional
            Filepath for saving the figure. Default is None, in that case no
            file is saved.

        Returns
        -------
        None.
        """
        fig, ax = plt.subplots(1, 1, figsize=(16 / 2.54, 12 / 2.54))

        ax.plot(self.T[self.t > max(self.t) / 2], self.M[self.t > max(self.t) / 2])

        ax.set_xlabel("Temperature in K")
        ax.set_ylabel("Magnetic moment in a.u.")
        ax.xaxis.set_inverted(True)
        ax.set_yticks([])
        fig.tight_layout()

        if filepath is not None:
            plt.savefig(filepath, dpi=300)

    def properties_to_txt(self, filepath, unit="T", sep="\t"):
        r"""
        Save all properties derived from the VSM-measurement to CSV-file.

        Parameters
        ----------
        filepath: STRING
            Fielpath to save the TXT-file to.
        unit: STRING
            Unit the remanence and coercivity are given in.
            Default is Tesla
        sep: STRING
            Seperator to be used during the pd.to_csv. Default is "\t"

        Returns
        -------
        None
        """
        if self.measurement == "M(H)":
            properties = {
                "Jr in " + unit: [self.remanence.to(unit).value],
                "iHc in " + unit: [self.coercivity.to(unit).value],
                r"BHmax in kJ/m^3": [self.BHmax.to(mu.kJ / mu.m**3).value],
                "S": [self.squareness],
            }
        elif self.measurement == "M(T)":
            properties = {"Tc in K": [self.Tc.value]}
        df = pd.DataFrame(properties)
        df.to_csv(filepath, sep=sep)

    def properties_to_json(self, filepath, unit="T"):
        """
        Save all properties derived from the VSM-measurement to json-file.

        Parameters
        ----------
        filepath: STRING
            Fielpath to save the TXT-file to.
        unit: STRING
            Unit the remanence and coercivity are given in.
            Default is Tesla

        Returns
        -------
        None
        """
        if self.measurement == "M(H)":
            properties = {
                "Jr in " + unit: [self.remanence.to(unit).value],
                "iHc in " + unit: [self.coercivity.to(unit).value],
                r"BHmax in kJ/m^3": [self.BHmax.to(mu.kJ / mu.m**3).value],
                "S": [self.squareness],
            }
        elif self.measurement == "M(T)":
            properties = {"Tc in K": [self.Tc.value]}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(properties, f, ensure_ascii=False, indent=4)

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
                "Jr in " + unit: [self.remanence.to(unit).value],
                "iHc in " + unit: [self.coercivity.to(unit).value],
                r"BHmax in kJ/m^3": [self.BHmax.to(mu.kJ / mu.m**3).value],
                "S": [self.squareness],
            }
        elif self.measurement == "M(T)":
            properties = {"Tc in K": [self.Tc.to("K").value]}
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
            f.create_dataset("Properties/Remanence", data=self.get_remanence(unit))
            f["Properties/Remanence"].attrs["unit"] = unit
            f.create_dataset("Properties/Coercivity", data=self.get_coercivity(unit))
            f["Properties/Coercivity"].attrs["unit"] = unit
            f.create_dataset("Properties/BHmax", data=self.get_remanence(unit))
            f["Properties/BHmax"].attrs["unit"] = "kJ/m^3"
            f.create_dataset("Properties/Squareness", data=self.get_squareness())
        elif self.measurement == "M(T)":
            f.create_dataset("Properties/Tc", data=self.get_Tc())
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
    filepath: STRING, optional
        Filepath for saving the figure. Default is None, in that case no file
        is saved.
    demag: BOOL, optional
        Boolean that determines if demagnetization curve is plotted as an
        inset next to hysteresis loop. Default is True.

    Returns
    -------
    None
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(16 / 2.54, 12 / 2.54))

    # plot hysteresis loops
    for i in range(len(data)):
        ax1.plot(data[i].H * mu_0, data[i].M * mu_0, label=labels[i])

    # format plot
    ax1.xaxis.set_major_locator(MultipleLocator(2))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax1.set_xlabel(r"$\mu_0 H_{int}$ in $T$")
    ax1.set_ylabel(r"$J$ in $T$")
    ax1.grid(visible=True, which="major", axis="both", linestyle=":", linewidth=1)
    ax1.grid(visible=True, which="minor", axis="both", linestyle=":", linewidth=0.5)
    ax1.legend()
    fig.tight_layout()
    plt.gca().set_prop_cycle(None)

    # plot inset of demagnetization curves
    if demag:
        ax2 = ax1.inset_axes([0.58, 0.15, 0.3, 0.5])
        for i in range(len(data)):
            ax2.plot(
                data[i].H[100:-100] * mu_0, data[i].M[100:-100] * mu_0, label=labels[i]
            )

        # find upper and lower border of plot, so that demagnetization
        # curve fits nicely
        coercmax = max([i.get_coercivity for i in data])
        remmax = max([i.get_remanence for i in data])
        Hmin, Jmax = None, None
        for i in np.arange(0, 5, 0.05):
            if Hmin is None and -i <= -coercmax - 0.02:
                Hmin = -i
            if Jmax is None and i >= remmax + 0.02:
                Jmax = i
            if Hmin is not None and Jmax is not None:
                break

        # format plot
        ax2.axis([Hmin, 0, 0, Jmax])
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.xaxis.set_major_locator(MultipleLocator(0.1))
        ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax2.yaxis.set_major_locator(MultipleLocator(0.1))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax2.set_xlabel(r"$\mu_0 H_{int}$ in $T$")
        ax2.set_ylabel(r"$J$ in $T$")
        ax2.grid(visible=True, which="major", axis="both", linestyle=":", linewidth=1)
        ax2.grid(visible=True, which="minor", axis="both", linestyle=":", linewidth=0.5)

    # save figure if filepath is given
    if filepath is not None:
        plt.savefig(filepath, dpi=300)
    return None


def mult_properties_to_txt(filepath, data, labels, unit="T", sep="\t"):
    r"""
    Save magnetic properties derived from the several VSM-measurements to
    CSV-file.

    Parameters
    ----------
    filepath: STRING
        Fielpath to save the TXT-file to.
    data : LIST
        List of VSM objects.
    labels : LIST
        List of labels (string) identifying the data rows.
        Has to be of same length as data.
    unit: STRING
        Unit the remanence and coercivity are given in.
        Default is Tesla
    sep: STRING
        Seperator to be used during the pd.to_csv. Default is "\t"

    Returns
    -------
    None
    """
    if all([i.measurement == "M(H)" for i in data]):
        properties = {
            "sample": labels,
            "Jr in " + unit: [i.get_remanence(unit) for i in data],
            "iHc in " + unit: [i.get_coercivity(unit) for i in data],
            r"BHmax in kJ/m^3": [i.get_BHmax() for i in data],
            "S": [i.get_squareness() for i in data],
        }
    elif all([i.measurement == "M(T)" for i in data]):
        properties = {"sample": labels, "Tc in K": [i.get_Tc() for i in data]}
    else:
        raise Exception("""Please only export a list of VSM measurements if
                        all of them are the same measurement type. This
                        does not seem to be the case.""")
    df = pd.DataFrame(properties)
    df.to_csv(filepath, sep=sep)


def mult_properties_to_json(filepath, data, labels, unit="T"):
    """
    Save magnetic properties derived from the several VSM-measurements to
    json-file.

    Parameters
    ----------
    filepath: STRING
        Fielpath to save the TXT-file to.
    data : LIST
        List of VSM objects.
    labels : LIST
        List of labels (string) identifying the data rows.
        Has to be of same length as data.
    unit: STRING
        Unit the remanence and coercivity are given in.
        Default is Tesla

    Returns
    -------
    None
    """
    if all([i.measurement == "M(H)" for i in data]):
        properties = {
            "sample": labels,
            "Jr in " + unit: [i.get_remanence(unit) for i in data],
            "iHc in " + unit: [i.get_coercivity(unit) for i in data],
            r"BHmax in kJ/m^3": [i.get_BHmax() for i in data],
            "S": [i.get_squareness() for i in data],
        }
    elif all([i.measurement == "M(T)" for i in data]):
        properties = {"sample": labels, "Tc in K": [i.get_Tc() for i in data]}
    else:
        raise Exception("""Please only export a list of VSM measurements if
                        all of them are the same measurement type. This
                        does not seem to be the case.""")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(properties, f, ensure_ascii=False, indent=4)


def diff(a, b):
    """
    Calculate the difference between value a and value b relative to
    value a in percent.

    Parameters
    ----------
    a : INT | FLOAT
        Numerical value.
    b : INT | FLOAT
        Numerical value.

    Returns
    -------
    FLOAT
        Difference of Value a and b in percent.
    """
    return (b - a) / a * 100


def droot(x, y):
    """
    Find root of a discrete dataset of x and y values.

    Parameters
    ----------
    x : ARRAY
        Dataset in horizontal axis, on which root point is located on.
    y : ARRAY
        Dataset in vertical axis.

    Returns
    -------
    FLOAT|ARRAY
        Array of root points. If only one is found, it's returned as float.
    """
    r = np.array([]) * x.unit
    # scan over whole range of values to find the two points where the
    # y-axis is crossed
    for i in range(len(x) - 1):
        # y values on left and right side of root will only be negative if
        # their product is negative
        if y[i] * y[i + 1] <= 0:
            # dataset between two points is assumed to be linear,
            # calculate linear equation to get exact interception point
            # with x-axis
            m = (y[i + 1] - y[i]) / (x[i + 1] - x[i])  # slope
            n = y[i] - m * x[i]  # y-intercept
            x0 = -n / m  # x-intercept
            # append x-intercepts as root to array
            r = np.append(r, x0)
    # Convert array of found roots to float if only one was found
    if np.shape(r) == (1,):
        r = r[0]
    return r
