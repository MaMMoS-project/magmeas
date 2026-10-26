"""Magnetometric Sweep class."""

import mammos_entity as me
import mammos_units as u


class Sweep(me.EntityCollection):
    """
    Class for importing, storing and using of magnetometric data aswell as derived
    parameters.
    """

    def read_custom(self, reader_func, *reader_args, **reader_kwargs):
        """
        Use a custom reader function and assign its output to self.
        This is intended to expand the file reading functionality of magmeas
        with custom functions for currently unsupported file types. It could
        however also be used to group several initialisation steps together,
        as all outputs of the custom reader function will be assigned to self
        without any checks. This doesn't only have to be measurement data like
        the ExternalMagneticField but could also be sample information, already
        calculated properties or anything else.

        The custom reader function `reader_func`is expected to return a single
        dictionary or mammos_entity.EntityCollection object. Its contents will
        be assigned to self without checks. It is strongly encouraged to only
        include mammos_entity.Entity objects in those output containers, since
        only those can later be used for magmeas-internal calculations.
        Please keep in mind, that some already present contents of self might be
        overwritten if the same keys are used.

        Also returns self, so the following works:
        `some_data = magmeas.Sweep().read_custom(some_reader_func, 'path/to/file.DAT')`

        Parameters
        ----------
        reader_func: FUNCTION
            The custom reader function which will return a dictionary or a
            mammos_entity.EntityCollection object with Entities that will be
            assigned to self.
        reader_args: optional
            Arguments which will be passed to reader_func. This could for
            example be the path to a file which is supposed to be read.
        reader_kwargs: optional
            Keyword arguments which will be passed to reader_func.
        """
        output = reader_func(*reader_args, **reader_kwargs)
        if isinstance(output, dict):
            for key, value in output.items():
                self[key] = value
        elif isinstance(output, me.EntityCollection):
            for key, value in output:
                self[key] = value
        else:
            raise TypeError(
                f"The reader_func returned an object of type {type(output)}. "
                + "This is currently not supported. Please only use reader "
                + "functions which return objects of type dictionary or "
                + "mammos_entity.EntityCollection."
            )
        return self

    def read_PPMS(self, filepath):
        """
        Read VSM-data from a Quantum Design .DAT file generated with Quantum
        Designs PPMS, DynaCool or VersaLab devices and assign it to self.
        Also returns self, so the following works:
        `some_data = magmeas.Sweep().read_PPMS('path/to/file.DAT')`

        Parameters
        ----------
        filepath: STR | PATH
            Path to Quantum Design .DAT file that data is supposed to be
            imported from

        Returns
        -------
        SWEEP-like
            Returns self, see above.
        """
        from magmeas.io import read_PPMS

        self.read_custom(read_PPMS, filepath)
        return self

    def get_entity_key(self, ontology_label):
        """Check, whether collection contains entities of the given label and
        return their respective keys. If no entities of this label can be found,
        the boolean False is returned.

        Parameters
        ----------
        ontology_label: STR
            Ontology label of the entities for which the keys are supposed to
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

    def calc_M(self, **kwargs):
        r"""
        Calculate Magnetization and MagneticPolarization and assign them to self
        with the respective keys 'M' and 'J'.
        For this, self must contain one (and only one) Entity which has the
        ontology_label 'MagneticMoment' and one of the following:

        - an Entity with the ontology_label 'Volume'
        - an Entity with the ontology_label 'Mass' AND one with 'Density'
        - an Entity with the ontology_label 'Length', containing three values
          (and only three!) which are assumed to be the side lenghts of a
          rectangular cuboid

        If more than one variant would be possible, only one will be chosen. The
        choice will fall on the option higher in the list above. If the thus
        chosen option is satisfied by more than one Entity contained in self,
        the calculation will be canceled as the ambiguity cannot be fixed
        automatically (e.g. two Entities have the ontology_label 'Volume').
        Optionally, the needed Entities from the list above can be given as
        keyword arguments of this function to provide them for this calculation.
        They will then be assigned to self and the same logic as before is
        followed. Assignments to self will happen whether the given Entities
        are used in the calculation or not. Already assigned Entities will be
        overwritten if the same key is used, which also applies to entities with
        the keys 'M' and 'J'.

        The following equations are used:
        .. math::
            M &= \frac{\mu}{V}\\
            J &= \mu _0 \cdot M\\
            V &= m \cdot \rho\\
            V &= l_1 \cdot l_2 \cdot l_3

        Parameters
        ----------
        kwargs: ENTITY, optional
            Keyword arguments that will be assigned to self and possibly
            contribute to the calculation of M and J if they are relevant.

        Returns
        -------
        None
        """
        moment_key = self.get_entity_key("MagneticMoment")
        if isinstance(moment_key, bool):
            raise AttributeError(
                "No entity with the ontology_label 'MagneticMoment' could be "
                + "found in self. Please provide a MagneticMoment first and try "
                + "again."
            )
        if isinstance(moment_key, list):
            raise AttributeError(
                "More than one entity with the ontology_label 'MagneticMoment' "
                + "could be found in self. Please provide only one. The entities "
                + f"found were {moment_key}."
            )

        for key, value in kwargs.items():
            if isinstance(value, me.Entity):
                self[key] = value
            else:
                raise TypeError(
                    f"The type of the kwarg ´{key}´ was {type(value)}, which is not "
                    + "allowed for this method. Please only pass kwargs of type "
                    + "mammos_entity.Entity to this method."
                )

        volume_key = self.get_entity_key("Volume")
        mass_key = self.get_entity_key("Mass")
        density_key = self.get_entity_key("Density")
        length_key = self.get_entity_key("Length")
        # self.get_entity_key only returns a str if the ontology_label was found
        # exactly once. Therefore, isinstance(key, str) is a sufficient check.
        if isinstance(volume_key, str):
            volume = self[volume_key].q
        elif isinstance(mass_key, str) and isinstance(density_key, str):
            volume = self[mass_key].q / self[density_key].q
        elif isinstance(length_key, str):
            if self[length_key].q.shape != (3,):
                raise AttributeError(
                    f"The entity ´{length_key}´ in self does have the ontology_label "
                    + "'Length', but not the right shape of (3,). The detected "
                    + f"shape was {self[length_key].q.shape}."
                )
            else:
                volume = (
                    self[length_key].q[0]
                    * self[length_key].q[1]
                    * self[length_key].q[2]
                )
        else:
            raise AttributeError(
                "No suitable sample information "
                + "(volume | (mass & density) | length) "
                + "could be found within self. Please provide sufficient information "
                + "for the calculation of M, see the docstring and try again."
            )

        self["M"] = me.Entity(
            "Magnetization",
            self[moment_key].q / volume,
            "A / m",
        )
        self["J"] = me.Entity(
            "MagneticPolarisation",
            self[moment_key].q / volume * u.constants.mu0,
            "T",
        )
