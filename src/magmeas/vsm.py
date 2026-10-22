"""VSM-like classes and functions using them."""

from pathlib import Path

import mammos_entity as me
import mammos_units as mu
import numpy as np
import pandas as pd
from mammos_analysis.demag import demag_cuboid

from magmeas._utils import rextract

mu.set_enabled_equivalencies(mu.magnetic_flux_field())


class VSM(me.EntityCollection):
    """
    Class for importing, storing and using of VSM-data aswell as derived
    parameters.
    """

    def __init__(self, filepath=None):
        super().__init__()
        if filepath is not None:
            self._path = Path(filepath)
            match self._path.suffix.lower():
                case ".dat":
                    self.load_qd(self._path)
                case _:
                    raise ValueError(
                        f"""
                        The given file extension is {self._path.suffix} and was 
                        not recognised. Please only use files in supported 
                        formats. See the documentation for more information.
                        """
                    )

    def load_qd(self, filepath):
        """
        Load VSM-data from a quantum systems .DAT file.

        Parameters
        ----------
        filepath: STR | PATH
            Path to Quantum Design .DAT file that data is supposed to be
            imported from

        Returns
        -------
        None

        """
        self._path = Path(filepath)

        err = """
              Sample parameters could not be read automatically.
              Please enter sample parameters correctly in the .DAT file like so:
              INFO,<mass in mg>,SAMPLE_MASS
              INFO,(<a>, <b>, <c>),SAMPLE_SIZE
              sample dimensions a, b and c in mm, c parallel to field
              """
        # open file and extract content as string
        with open(self._path, encoding="cp1252") as f:
            s = f.read()
        # figure out in which line actual CSV-like data starts for later import
        head_length = s[: s.find("[Data]")].count("\n") + 1

        # check if sample mass can be read from .DAT file
        try:
            mass = mu.Quantity(float(rextract(s, "INFO,", ",SAMPLE_MASS")), "mg")
        except ValueError:
            raise Exception(err) from None
        # check if sample dimensions can be read from .DAT file
        try:
            dim = rextract(s, "INFO,(", "),SAMPLE_SIZE").split(",")
            dim = mu.Quantity(np.array([float(f) for f in dim]), "mm")
        except ValueError:
            raise Exception(err) from None

        # calculate density from sample mass and dimensions
        density = mass / (np.prod(dim.value) * dim.unit**3)
        # calculate demagnetisation factor
        self.D = demag_cuboid(*dim)
        # extract only demag factor needed for the calculations done here
        self.D = me.Entity("DemagnetizingFactor", self.D.q[-1])
        # import measurement data
        df = pd.read_csv(self._path, skiprows=head_length, encoding="cp1252")
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
            m = me.Entity("MagneticMoment", df["Moment (emu)"], "erg / G")
        elif m_unit == "Am2":
            m = me.Entity("MagneticMoment", df["Moment (Am2)"], "A m2")
        else:
            raise ValueError(
                "Unit of magnetic moment seems to be neither emu,"
                " nor Am2. Other units are not supported."
            )
        # extract external magnetic field
        eH = me.Entity("ExternalMagneticField", df["Magnetic Field (Oe)"], "Oe")
        # calculate magnetization
        M = me.Entity("Magnetization", m.q / mass * density)
        # calculate internal magnetic field
        H = me.Entity("InternalMagneticField", eH.q - self.D.q * M.q)
        # extract absolute temperature
        T = me.Entity("Temperature", np.array(df["Temperature (K)"]) * mu.K)
        # extract time stamp
        t = me.Entity("Time", np.array(df["Time Stamp (sec)"]) * mu.s)

        # test datapoints for missing values (where value is nan)
        # H is derived from H_ext, and m so this is sufficient to test both
        nanfilter = ~np.isnan(H.q) * ~np.isnan(T.q) * ~np.isnan(t.q)
        # delete all datapoints where H_ext, m, T or t are nan
        H = me.Entity(H.ontology_label, H.q.to("A/m")[nanfilter])
        H_ext = me.Entity(eH.ontology_label, eH.q.to("A/m")[nanfilter])
        M = me.Entity(M.ontology_label, M.q.to("A/m")[nanfilter])
        m = me.Entity(m.ontology_label, m.q.to("A m2")[nanfilter])
        T = me.Entity(T.ontology_label, T.q[nanfilter])
        # convert time stamp to time since measurement start
        t = me.Entity(t.ontology_label, t.q[nanfilter] - t.q[nanfilter][0])
        # assign Entities to self as EntityCollection
        self.measurement_data = me.EntityCollection(
            "Data derived from the VSM-measurement.",
            H=H,
            H_ext=H_ext,
            M=M,
            m=m,
            T=T,
            t=t,
        )

    @property
    def path(self) -> Path:
        """Path to the file which was used in the object generation."""
        return self._path

    @property
    def H(self) -> me.Entity:
        """
        The internal magnetic field calculated from the measurement data.
        Calculated according to:
            .. math:: H_{int} = H_{ext} - D \\cdot M
        """
        return self.measurement_data.H

    @property
    def H_ext(self) -> me.Entity:
        """The external magnetic field as read from the measurement file."""
        return self.measurement_data.H_ext

    @property
    def M(self) -> me.Entity:
        """
        The magnetisation calculated from the measurement data. Calculated
        according to:
            .. math:: M = \\frac{m}{V}
        with m being the magnetic moment and V being the volume.
        """
        return self.measurement_data.M

    @property
    def m(self) -> mu.Quantity:
        """The magnetic moment as read from the measurement file."""
        return self.measurement_data.m

    @property
    def T(self) -> me.Entity:
        """The thermodynamic temperature as read from the measurement file."""
        return self.measurement_data.T

    @property
    def t(self) -> me.Entity:
        """The time since the start of the measurement."""
        return self.measurement_data.t

    def to_csv(self, filename):
        """
        Write measurement data to CSV-file. See `mammos_entity.EntityCollection.to_csv`
        for more details.

        filename: STR | PATH
            Name of the generated file.
        """
        self.measurement_data.to_csv(filename)
