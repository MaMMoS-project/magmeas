"""VSM-like classes and functions using them."""

import mammos_entity as me


class VSM(me.EntityCollection):
    """
    Class for importing, storing and using of VSM-data aswell as derived
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
        `vsm = magmeas.VSM().read_custom(some_reader_func, 'path/to/file.DAT')`

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
        `vsm = magmeas.VSM().read_PPMS('path/to/file.DAT')`

        Parameters
        ----------
        filepath: STR | PATH
            Path to Quantum Design .DAT file that data is supposed to be
            imported from

        Returns
        -------
        VSM-like
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
