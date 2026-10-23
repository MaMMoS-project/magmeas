"""VSM-like classes and functions using them."""

import mammos_entity as me
import numpy as np
import pandas as pd


class VSM(me.EntityCollection):
    """
    Class for importing, storing and using of VSM-data aswell as derived
    parameters.
    """

    def read_PPMS(self, filepath):
        """
        Load VSM-data from a quantum systems .DAT file generated with Quantum
        Designs PPMS, DynaCool or VersaLab devices.

        Parameters
        ----------
        filepath: STR | PATH
            Path to Quantum Design .DAT file that data is supposed to be
            imported from

        Returns
        -------
        None

        """
        # open file and extract content as string
        with open(filepath, encoding="cp1252") as f:
            s = f.read()
        # figure out in which line actual CSV-like data starts for later import
        head_length = s[: s.find("[Data]")].count("\n") + 1
        # import measurement data
        df = pd.read_csv(filepath, skiprows=head_length, encoding="cp1252")
        # extract magnetic moment
        # first, let's find the unit of the magnetic moment
        for col in df.columns:
            start_idx = col.find("Moment (")
            if start_idx != -1:
                start_idx = col.find("(") + 1
                end_idx = col.find(")")
                mu_unit = col[start_idx:end_idx]
        # if the unit is declared as 'emu', we need to specify it explicitly
        # since emu is ambiguous and thus not included in mammos-units
        if mu_unit == "emu":
            mu = me.Entity("MagneticMoment", df["Moment (emu)"], "erg / G")
        elif mu_unit == "Am2":
            mu = me.Entity("MagneticMoment", df["Moment (Am2)"], "A m2")
        else:
            raise ValueError(
                f"Unit of magnetic moment was {mu_unit} which is not compatible"
                + ", as only emu and Am2 are supported at the moment. "
                + "Please add the measurement data manually."
            )
        # extract external magnetic field
        H_ext = me.Entity("ExternalMagneticField", df["Magnetic Field (Oe)"], "Oe")
        # extract absolute temperature
        T = me.Entity("ThermodynamicTemperature", df["Temperature (K)"], "K")
        # extract time stamp
        t = me.Entity("Time", df["Time Stamp (sec)"], "s")

        # test datapoints for missing values (where value is nan)
        filter = ~np.isnan(
            np.column_stack([H_ext.value, mu.value, T.value, t.value])
        ).any(axis=1)
        # delete all datapoints where H_ext, m, T or t are nan
        H_ext = me.Entity(H_ext.ontology_label, H_ext, "A/m")[filter]
        mu = me.Entity(mu.ontology_label, mu, "A m2")[filter]
        T = me.Entity(T.ontology_label, T, "K")[filter]
        # convert time stamp to time since measurement start
        t = me.Entity(t.ontology_label, t.q - t.q[0], "s")[filter]
        # assign Entities to self
        self.H_ext = H_ext
        self.mu = mu
        self.T = T
        self.t = t
        # return self, which enables convenient initialisation in one step
        # example: vsm = magmeas.VSM().read_PPMS("path/to/file.DAT")
        return self

    def get_entity_key(self, ontology_label):
        """Check, whether collection contains entities of the given label and
        return their respective keys. If no entities of this label can be found,
        the boolean False is returned.

        Parameters
        ----------
        ontology_label: STR
            Ontology label of the entities for which their keys are supposed to
            be returned.

        Returns
        -------
        STR | LIST[STR] | BOOL
            False if entities with the given ontology label could not be found,
            the key of the entity that had the specified label if only one was
            found or a list of all keys for which the respective entity has the
            specified ontology label.
        """
        entities = {key: entity.ontology_label for key, entity in self}
        labels = set(entities.values())
        if ontology_label not in labels:
            return False
        results = {label: [] for label in labels}
        for key, value in entities.items():
            results[value].append(key)
        if len(results[ontology_label]) == 1:
            return results[ontology_label][0]
        else:
            return results[ontology_label]
