"""Magnetometric Sweep class."""

import mammos_entity as me
import mammos_units as u
import matplotlib.pyplot as plt
from mammos_analysis.demag import demag_cuboid

from magmeas._utils import _arg_processing


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

            some_data = magmeas.Sweep().read_custom(some_func, 'path/to/file.DAT')

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

            some_data = magmeas.Sweep().read_PPMS('path/to/file.DAT')

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
        then None is returned.

        Parameters
        ----------
        ontology_label: STR
            Ontology label of the entities for which the keys are supposed to
            be returned.

        Returns
        -------
        STR | LIST[STR] | NONE
            None if entities with the given ontology label could not be found,
            the key of the entity that had the specified label if only one was
            found or a list of all keys for which the respective entity has the
            specified ontology label.
        """
        entities = {key: entity.ontology_label for key, entity in self}
        labels = set(entities.values())
        if ontology_label not in labels:
            return None
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
        if moment_key is None:
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

    def corr_demag(self, **kwargs):
        r"""
        Correct the ExternalMagneticField for the demagnetising field created by
        the samples own Magnetization. This yields the InternalMagneticField,
        which will be assigned to self with the key 'H'.
        For this, self must contain one (and only one) Entity which has the
        ontology_label 'ExternalMagneticField', another one with 'Magnetization'
        and one of the following:

        - an Entity with the ontology_label 'DemagnetizingFactor'
        - an Entity with the ontology_label 'Length', containing three values
          (and only three!) which are assumed to be the side lenghts of a
          rectangular cuboid. The FIRST value in this Entity is assumed to be
          parallel to the direction of the ExternalMagneticField.
          The function mammos_analysis.demag.rectangular_cuboid will be used to
          calculate the DemagnetizingFactor of this rectangular cuboid. Doing so
          manually and only passing the relevant value as its own Entity with
          the ontology_label 'DemagnetizingFactor' like above is recommended.

        If more than one variant would be possible, only one will be chosen. The
        choice will fall on the option higher in the list above. If the thus
        chosen option is satisfied by more than one Entity contained in self,
        the calculation will be canceled as the ambiguity cannot be fixed
        automatically (e.g. two Entities have the ontology_label 'Length').
        Optionally, the needed Entities from the list above can be given as
        keyword arguments of this function to provide them for this calculation.
        They will then be assigned to self and the same logic as before is
        followed. Assignments to self will happen whether the given Entities
        are used in the calculation or not. Already assigned Entities will be
        overwritten if the same key is used, which also applies to entities with
        the key 'H'.

        The following equations are used:

        .. math::

            H_{int} = H_{ext} - D \cdot M

        Parameters
        ----------
        kwargs: ENTITY, optional
            Keyword arguments that will be assigned to self and possibly
            contribute to the calculation of H if they are relevant.

        Returns
        -------
        None
        """
        magnetisation_key = self.get_entity_key("Magnetization")
        ext_field_key = self.get_entity_key("ExternalMagneticField")
        if magnetisation_key is None or ext_field_key is None:
            raise AttributeError(
                "Either an Entity with the ontology_label 'Magnetization' or "
                + " 'ExternalMagneticField' could not be found in self. "
                + "Please provide both first and try again."
            )
        if isinstance(magnetisation_key, list) or isinstance(ext_field_key, list):
            raise AttributeError(
                "More than one entity with the ontology_label 'Magnetization' "
                + "or 'ExternalMagneticField' could be found in self. "
                + "Please provide only one. The entities found were "
                + f"{magnetisation_key} and {ext_field_key}."
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

        demag_key = self.get_entity_key("DemagnetizingFactor")
        length_key = self.get_entity_key("Length")
        # self.get_entity_key only returns a str if the ontology_label was found
        # exactly once. Therefore, isinstance(key, str) is a sufficient check.
        if isinstance(demag_key, str):
            demag_factor = self[demag_key]
        elif isinstance(length_key, str):
            if self[length_key].q.shape != (3,):
                raise AttributeError(
                    f"The entity ´{length_key}´ in self does have the ontology_label "
                    + "'Length', but not the right shape of (3,). The detected "
                    + f"shape was {self[length_key].q.shape}."
                )
            else:
                demag_factor = demag_cuboid(*self[length_key].q)[0]
        else:
            raise AttributeError(
                "No suitable sample information (DemagnetizingFactor | Length)"
                + "could be found within self. Please provide sufficient information "
                + "for the calculation of H, see the docstring and try again."
            )

        self["H"] = me.Entity(
            "InternalMagneticField",
            self[ext_field_key].q - self[magnetisation_key].q * demag_factor.q,
            "A/m",
        )

    def plot(
        self,
        *args,
        filepath=None,
        bounds=(None, None),
        fig_ax=None,
        **kwargs,
    ):
        """
        Plot arguments and save figure if a filepath is given.

        Parameters
        ----------
        args: STR | ENTITY | QUANTITY | ARRAY
            This determines the values to be plotted. Can only be comprised of
            one or two values. In the case of a string, it is first attempted
            to use it as a key for the values in self. If the string is not a
            key in self, then it is assumed to be an ontology_label and the
            entitity in self with the fitting ontology_label will be used. If
            more than one entities have a fitting ontology_label, this method
            fails.
            In the case of an Entity, a Quantity or an array, they are attempted
            to be plotted directly.
        filepath: STR | PATH, optional
            Filepath for saving the figure. Default is None, in that case no
            file is saved and the generated figure as well as all axes objects
            are returned.
        bounds: TUPLE(NONE | INT), optional
            Indices (start, end) at which the data to be plotted will be cut
            off. Use None to plot until the respective start or end.
            Default is (None, None).
        fig_ax: TUPLE(matplotlib.figure.Figure, matplotlib.axes.Axes), optional
            Tuple of figure and axes the data should be plotted to. If None is
            given they will be generated dynamically. Default is None.
        kwargs:
            Keyword arguments to change the appearance of plots.
            See documentation of matplotlib.pyplot.plot for more information.

        Returns
        -------
        None | (matplotlib.Figure, matplotlib.Axes)
            Return figure and axes object if filepath is not specified.
        """
        start, end = bounds
        if fig_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax

        if len(args) == 0:
            raise Exception("No values to plot were given, nothing to do.")
        elif len(args) == 1:
            y = _arg_processing(self, args[0])
            if isinstance(y, me.Entity):
                ax.plot(y.q[start:end], **kwargs)
                ax.set_ylabel(y.axis_label)
            elif isinstance(y, u.Quantity):
                ax.plot(y[start:end], **kwargs)
                ax.set_ylabel("(" + str(y.unit) + ")")
            else:
                ax.plot(y[start:end], **kwargs)
        elif len(args) == 2:
            x, y = (_arg_processing(self, arg) for arg in args)
            if isinstance(x, me.Entity):
                ax.set_xlabel(x.axis_label)
                x = x.q
            elif isinstance(x, u.Quantity):
                ax.set_xlabel("(" + str(x.unit) + ")")
            if isinstance(y, me.Entity):
                ax.set_ylabel(y.axis_label)
                y = y.q
            elif isinstance(y, u.Quantity):
                ax.set_ylabel("(" + str(y.unit) + ")")
            ax.plot(x[start:end], y[start:end], **kwargs)
        else:
            raise ValueError(
                f"A number of {len(args)} args was given. Please use at most two."
            )

        # save figure if filepath is given
        if filepath is not None:
            fig.savefig(filepath)
        if filepath is None:
            return fig, ax
