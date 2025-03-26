# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:48:00 2024

@author: jw14
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# %%

mu_0 = np.pi * 4e-7


class VSM:

    """
    Class for importing, storing and using of VSM-data aswell as derived
    parameters.

    Attributes
    ----------
    H: NUMPY-ARRAY
        Magnetic field strength in A/m
    M: NUMPY-ARRAY
        Magnetization in A/m
    T: NUMPY-ARRAY
       Absolute temperature in K

    Methods
    -------
    load_qd()
        Load VSM-data from a quantum design .DAT file
    get_saturization()
        Getter function for the saturation polarization or magnetization.
        Calculated using the approach to saturation method.
    get_remanence()
        Getter function for the remanent polarization or magnetization.
    get_coercivity()
        Getter function for the coercivity
    get_BHmax()
        Getter function for the maximum energy product BHmax
    get_squareness()
        Getter function for the squareness
    plot()
        Plots hysteresis loop, optionally with inset of demagnetisation curve,
        optionally saves as png
    properties_to_txt()
        Saves all properties derived from VSM-measurement to CSV-file.
    """

    def __init__(self, datfile):

        self.load_qd(datfile)

    def demag_prism(self, a, b, c):
        """
        Calculates demagnetization factor Dz for a rectangular prism with
        dimensions a, b, and c where c is assumed to be the axis along which
        the prism was magnetized.\n
        Copied from
        https://rmlmcfadden.github.io/bnmr/technical-information/calculators/\n
        Equation from A. Aharoni, J. Appl. Phys. 83, 3422 (1998).
        https://doi.org/10.1063/1.367113
        Eq. (1) - see Fig. 1 for_abc coordinate system

        Parameters
        ----------
        a: FLOAT
            One side length of rectangular prism, perpendicular to
            magnetization direction.
        b: FLOAT
            One side length of rectangular prism, perpendicular to
            magnetization direction.
        c: FLOAT
            One side length of rectangular prism, parallel to magnetization
            direction.

        Returns
        -------
        D: FLOAT
            Demagnetization factor along axis of magnetization (c-axis).
        """

        # the expression takes input as half of the semi-axes
        a = 0.5 * a
        b = 0.5 * b
        c = 0.5 * c     # c is || axis along which the prism was magnetized
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
            + 2 * np.arctan2(ab, c * r_abc)
            + (a2 * a + b2 * b - 2 * c2 * c) / (3 * abc)
            + ((a2 + b2 - 2 * c2) / (3 * abc)) * r_abc
            + (c / ab) * (r_ac + r_bc)
            - (r_ab * r_ab * r_ab + r_bc * r_bc *
               r_bc + r_ac * r_ac * r_ac) / (3 * abc)
        )
        # divide out the factor of pi
        D = pi_Dz / np.pi
        return D

    def calc_saturation(self):
        """
        Calculates saturation magnetization or polarization, using the approach
        to saturation method. Accounts for high field susceptibility.

        Parameters
        ----------
        unit: STRING, optional
            Unit the saturation is supposed to be returned in. Default is Tesla

        Returns
        -------
        saturation: len 2 TUPLE
            Tuple of value and unit of saturation magnetization/polarization
            (depending on given unit)
        """

        x = self.H
        y = self.M

        H_max = np.max(x)  # maximum magnetic field

        # get indices of all points where H_max - 1 MA/m < H <= H_max
        # assumed to be nearly linear region of highest fields
        # note how this does not discriminate between magnetization and
        # demagnetization, also not between initial and final saturation
        # due to this the calculated saturation will be somewhat of an
        # average value
        filt_ind = np.where(x > H_max - 1e6)

        # linear regression of high-field region to extract
        # high-field susceptibility
        linreg1 = linregress(x[filt_ind], y[filt_ind])

        # substract high-field susceptibility contribution from magnetization
        # and find intercept of another linear regression of 1/H over M
        # this lets H go towards infinity for 1/H = 0 (intercept of regression)
        # and is thus more accurate for a saturation magnetization
        # if no high field susceptibility is contributing to the magnetization,
        # then there is also nothing getting subtracted
        linreg2 = linregress(
            1/x[filt_ind], y[filt_ind] - x[filt_ind] * linreg1.slope)
        M_sat = linreg2.intercept

        a = {'A/m': M_sat,
             'kA/m': M_sat*1e-3,
             'T': M_sat*mu_0,
             'mT': M_sat*mu_0*1e3}
        return a

    def calc_remanence(self):
        """
        Extracts remanent magnetization or polarization from hysteresis loop

        Parameters
        ----------
        unit: STRING, optional
            Unit the remanence is supposed to be returned in. Default is Tesla.

        Returns
        -------
        remanence: len 2 TUPLE
            Tuple of value and unit of remanent magnetization/polarization
            (depending on given unit)
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
        # save the remanence in different units to be called directly without
        # need for later conversion
        a = {'A/m': a,
             'kA/m': a*1e-3,
             'T': a*mu_0,
             'mT': a*mu_0*1e3}
        return a

    def calc_coercivity(self):
        """
        Extracts intrinsic Coercivity from hysteresis loop

        Parameters
        ----------
        unit: STRING, optional
            Unit the remanence is supposed to be returned in. Default is Tesla.

        Returns
        -------
        coercivity: len 2 TUPLE
            Tuple of value and unit of intrinsic coercivity
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
        a = {'A/m': a,
             'kA/m': a*1e-3,
             'T': a*mu_0,
             'mT': a*mu_0*1e3}
        return a

    def calc_BHmax(self):
        """
        Extracts maximum Energy product from demagnetization curve

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
        return np.min(BH) * -1e-3

    def calc_squareness(self):
        """
        Calculates squareness of demagnetization curve

        Parameters
        ----------
        None

        Returns
        -------
        S: FLOAT
        Squareness (dimensionless)
        """
        # value that magnetization is supposed to have at knee-point
        Mk = 0.9 * self.get_remanence(unit='A/m')
        # find intersections of Hysteresis loop with M=Mk
        a = droot(self.H, self.M - Mk)
        a = np.abs(a)  # get absolute values
        a = a[1]  # second root should be knee field strength
        S = a / self.get_coercivity(unit='A/m')
        return S

    def load_qd(self, datfile, unit='T'):
        """
        Load VSM-data from a quantum systems .DAT file.

        Parameters
        ----------
        datfile: STRING
            Path to quantum systems .DAT file that data is supposed to be
            imported from
        unit: STRING, optional
            Unit the magnetic properties are supposed to be saved in.
            Default is Tesla.
        Returns
        -------
        None

        """

        def rextract(string, startsub, endsub):
            endind = string.index(endsub)
            startind = string.rindex(startsub, 0, endind)
            return string[startind + len(startsub):endind]

        with open(datfile, 'rb') as f:
            s = str(f.read(-1))
        mass = float(rextract(s, 'INFO,', ',SAMPLE_MASS')) * 1e-3  # mass in g
        dim = rextract(s, 'INFO,(', '),SAMPLE_SIZE').split(',')
        dim = [float(f) for f in dim]
        density = mass / np.prod(dim) / 1e-3
        D = self.demag_prism(dim[0], dim[1], dim[2])

        # import measurement data
        df = pd.read_csv(datfile, skiprows=34, encoding='cp1252')
        m = np.array(df['Moment (emu)'])  # save moments as np.array
        # save field as np.array and convert from Oe to A/m
        H = np.array(df['Magnetic Field (Oe)']) * 1E-4 / mu_0
        M = m / mass * density  # calc magnetisation in kA/m
        M = M * 1E3  # convert magnetisation from kA/m to A/m
        H = H - D * M  # correct field for demagnetisation
        # save absolute temperature as np.array
        T = np.array(df['Temperature (K)'])

        # test datapoints for missing values (where value is nan)
        nanfilter = [True]*len(H)
        for i in range(len(H)):
            nanfilter[i] = ~np.isnan(H[i]) and ~np.isnan(
                M[i]) and ~np.isnan(T[i])
        # delete all datapoints where any of H, M or T ar nan
        if ~np.all(nanfilter):
            H = H[nanfilter]
            M = M[nanfilter]
            T = T[nanfilter]
        self.H = H
        self.M = M
        self.T = T

        self._saturation = self.calc_saturation()
        self._remanence = self.calc_remanence()
        self._coercivity = self.calc_coercivity()
        self._BHmax = self.calc_BHmax()
        self._squareness = self.calc_squareness()

    def get_saturation(self, unit='T'):
        """
        Getter function for saturation. Depending on the given unit this will
        be the saturation magnetization or saturation polarization.

        Parameters
        ----------
        unit : STR, optional
            Unit the value of the saturation is given in. The default is 'T'.

        Returns
        -------
        FLOAT
            Returns value of the saturation in the specified unit.
        """
        return self._saturation[unit]

    def get_remanence(self, unit='T'):
        """
        Getter function for remanence. Depending on the given unit this will
        be the remanent magnetization or remanent polarization.

        Parameters
        ----------
        unit : STR, optional
            Unit the value of the remanence is given in. The default is 'T'.

        Returns
        -------
        FLOAT
            Returns value of the remanence in the specified unit.
        """
        return self._remanence[unit]

    def get_coercivity(self, unit='T'):
        """
        Getter function for coercivity. Depending on the given unit this might
        return a product of coercivity and the vacuum permeability.

        Parameters
        ----------
        unit : STR, optional
            Unit the value of the coercivity is given in. The default is 'T'.

        Returns
        -------
        FLOAT
            Returns value of the coercivity in the specified unit.
        """
        return self._coercivity[unit]

    def get_BHmax(self):
        """
        Getter function for saturation. Implemented for consistency.

        Parameters
        ----------
        NONE

        -------
        FLOAT
            Returns value of BHmax in kJ/m**3
        """
        return self._BHmax

    def get_squareness(self):
        """
        Getter function for squareness. Implemented for consistency.

        Parameters
        ----------
        NONE

        Returns
        -------
        FLOAT
            Returns value of squareness
        """
        return self._squareness

    def plot(self, filepath=None, demag=True, label=None):
        """
        Plots hysteresis loop, optionally with inset of demagnetization curve
        and saves figure if a filepath is given.

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

        H = self.H * mu_0  # converts H from A/m to Tesla
        M = self.M * mu_0  # converts M from A/m to Tesla

        fig, ax1 = plt.subplots(1, 1, figsize=(16/2.54, 12/2.54))
        ax1.plot(H, M, label=label)

        # find upper and lower border of plot, so that hysteresis loop fits
        Jmax = None
        for i in np.arange(0, 5, 0.1):
            if Jmax is None:
                if i >= self.get_saturation('T') + 0.02:
                    Jmax = i
                    break

        # format plot
        ax1.axis([-15, 15, -Jmax, Jmax])
        ax1.xaxis.set_major_locator(MultipleLocator(2))
        ax1.xaxis.set_minor_locator(MultipleLocator(.5))
        ax1.yaxis.set_major_locator(MultipleLocator(0.2))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax1.set_xlabel(r'$\mu_0 H_{int}$ in $T$')
        ax1.set_ylabel(r'$J$ in $T$')
        ax1.grid(visible=True, which='major',
                 axis='both', linestyle=':', linewidth=1)
        ax1.grid(visible=True, which='minor', axis='both',
                 linestyle=':', linewidth=0.5)
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
                if Hmin is None:
                    if -i <= -self.get_coercivity('T') - 0.02:
                        Hmin = -i
                if Jmax is None:
                    if i >= self.get_remanence('T') + 0.02:
                        Jmax = i
                if Hmin is not None and Jmax is not None:
                    break

            # format plot
            ax2.axis([Hmin, 0, 0, Jmax])
            ax2.yaxis.set_label_position('right')
            ax2.yaxis.tick_right()
            ax2.xaxis.set_major_locator(MultipleLocator(0.1))
            ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
            ax2.yaxis.set_major_locator(MultipleLocator(0.1))
            ax2.yaxis.set_minor_locator(MultipleLocator(0.05))
            ax2.set_xlabel(r'$\mu_0 H_{int}$ in $T$')
            ax2.set_ylabel(r'$J$ in $T$')
            ax2.grid(visible=True, which='major',
                     axis='both', linestyle=':', linewidth=1)
            ax2.grid(visible=True, which='minor', axis='both',
                     linestyle=':', linewidth=0.5)
            fig.tight_layout()

        # save figure if filepath is given
        if filepath is not None:
            plt.savefig(filepath, dpi=300)

    def properties_to_txt(self, filepath, unit='T', sep="\t"):
        """
        Saves all properties derived from the VSM-measurement to CSV-file.

        Parameters
        ----------
        filepath: STRING
            Fielpath to save the TXT-file to.
        unit: STRING
            Unit the saturation, remanence and coercivity are given in.
            Default is Tesla
        sep: STRING
            Seperator to be used during the pd.to_csv. Default is "\t"

        Returns
        -------
        None
        """

        df = pd.DataFrame(
            {
                "Js in "+unit: [self.get_saturation(unit)],
                "Jr in "+unit: [self.get_remanence(unit)],
                "iHc in "+unit: [self.get_coercivity(unit)],
                r"BHmax in kJ/m^3": [self.get_BHmax()],
                "S": [self.get_squareness()]
            }
        )
        df.to_csv(filepath, sep=sep)


def plot_multiple_VSM(data, labels, filepath=None, demag=True):
    """
    Plots hysteresis loops and optionally demagnetization curves of
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

    fig, ax1 = plt.subplots(1, 1, figsize=(16/2.54, 12/2.54))

    # plot hysteresis loops
    for i in range(len(data)):
        ax1.plot(data[i].H * mu_0, data[i].M * mu_0, label=labels[i])

    # find upper and lower border of plot, so that hysteresis loops fits
    Jmax = None
    satmax = max([i.get_saturation('T') for i in data])
    for i in np.arange(0, 5, 0.1):
        if Jmax is None:
            if i >= satmax + 0.02:
                Jmax = i + 0.05
                break

    # format plot
    ax1.axis([-15, 15, -Jmax, Jmax])
    ax1.xaxis.set_major_locator(MultipleLocator(2))
    ax1.xaxis.set_minor_locator(MultipleLocator(.5))
    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax1.set_xlabel(r'$\mu_0 H_{int}$ in $T$')
    ax1.set_ylabel(r'$J$ in $T$')
    ax1.grid(visible=True, which='major',
             axis='both', linestyle=':', linewidth=1)
    ax1.grid(visible=True, which='minor', axis='both',
             linestyle=':', linewidth=0.5)
    ax1.legend()
    fig.tight_layout()
    plt.gca().set_prop_cycle(None)

    # plot inset of demagnetization curves
    if demag:
        ax2 = ax1.inset_axes([0.58, 0.15, 0.3, 0.5])
        for i in range(len(data)):
            ax2.plot(data[i].H[100:-100] * mu_0,
                     data[i].M[100:-100] * mu_0, label=labels[i])

        # find upper and lower border of plot, so that demagnetization
        # curve fits nicely
        coercmax = max([i.get_coercivity for i in data])
        remmax = max([i.get_remanence for i in data])
        Hmin, Jmax = None, None
        for i in np.arange(0, 5, 0.05):
            if Hmin is None:
                if -i <= -coercmax - 0.02:
                    Hmin = -i
            if Jmax is None:
                if i >= remmax + 0.02:
                    Jmax = i
            if Hmin is not None and Jmax is not None:
                break

        # format plot
        ax2.axis([Hmin, 0, 0, Jmax])
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.tick_right()
        ax2.xaxis.set_major_locator(MultipleLocator(0.1))
        ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax2.yaxis.set_major_locator(MultipleLocator(0.1))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax2.set_xlabel(r'$\mu_0 H_{int}$ in $T$')
        ax2.set_ylabel(r'$J$ in $T$')
        ax2.grid(visible=True, which='major',
                 axis='both', linestyle=':', linewidth=1)
        ax2.grid(visible=True, which='minor', axis='both',
                 linestyle=':', linewidth=0.5)

    # save figure if filepath is given
    if filepath is not None:
        plt.savefig(filepath, dpi=300)
    return None


def mult_properties_to_txt(filepath, data, labels, unit='T', sep='\t'):
    """
    Saves magnetic properties derived from the several VSM-measurements to
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
        Unit the saturation, remanence and coercivity are given in.
        Default is Tesla
    sep: STRING
        Seperator to be used during the pd.to_csv. Default is "\t"

    Returns
    -------
    None
    """

    df = pd.DataFrame(
        {
            "sample": labels,
            "Js in "+unit: [i.get_saturation(unit)[0] for i in data],
            "Jr in "+unit: [i.get_remanence(unit)[0] for i in data],
            "iHc in "+unit: [i.get_coercivity(unit)[0] for i in data],
            r"BHmax in kJ/m^3": [i.get_BHmax for i in data],
            "S": [i.get_squareness for i in data]
        }
    )
    df.to_csv(filepath, sep=sep)


def diff(a, b):
    """
    Calculates the difference between value a and value b relative to
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
    return (b - a) / a


def droot(x, y):
    """
    Finding root of a discrete dataset of x and y values.

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
    r = np.array([])
    # scan over whole range of values to find the two points where the
    # y-axis is crossed
    for i in range(len(x)-1):
        # y values on left and right side of root will only be negative if
        # their product is negative
        if y[i] * y[i+1] <= 0:
            # dataset between two points is assumed to be linear,
            # calculate linear equation to get exact interception point
            # with x-axis
            m = (y[i+1]-y[i])/(x[i+1]-x[i])  # slope
            n = y[i] - m * x[i]  # y-intercept
            x0 = -n / m  # x-intercept
            # append x-intercepts as root to array
            r = np.append(r, x0)
    # Convert array of found roots to float if only one was found
    if np.shape(r) == (1,):
        r = r[0]
    return r
