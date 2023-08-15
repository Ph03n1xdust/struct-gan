from typing import Any, Callable, Sequence

import jax
import numpy as np
from ase import Atoms

from structgan.model import Generator


def get_max_R_CUT(atoms_arr: Sequence[Atoms]):
    """Calculates the maximum possible cutoff radius for a list of structures
        with periodic boundary conditions.

    Args:
        atoms_arr: Sequence of Atoms objects corresponding to the structures.

    Returns:
        The maximum possible cutoff radius for the specific set of structures.
    """
    possible_r_cuts = []

    for atoms in atoms_arr:
        volume = atoms.get_volume()

        side_areas = []
        for a in atoms.cell[:]:
            for b in atoms.cell[:]:
                side_areas.append(np.sum(np.cross(a, b) ** 2) ** 0.5)
        max_area = np.max(side_areas)
        curr_r_cut = volume / max_area / 2
        possible_r_cuts.append(curr_r_cut)

    return np.min(possible_r_cuts)


def create_generate_descriptor(descriptor_method: Callable):
    """Creates a function to generate descriptors of a single structure.

    Args:
        descriptor_method: A function used to generate descriptors.

    Returns:
        A function which generates descriptors of the chosen atoms in a given
        structure. It uses five arguments, the position of all atoms, the types
        of all atoms, the positions of the chosen atoms, the types of the
        chosen atoms and the unit cell respectively.
    """

    def generate_descriptor(allpos, alltype, pos, type, cell):
        """Generates descriptors of chosen atoms in a single structure.

        Args:
            allpos: Positions of all atoms with (n_atoms, n_dimension) shape.
            alltype: Types of all atoms indexed from 0 with (n_atoms,) shape.
                Atoms with negative types are ignored.
            pos: Positions of chosen atoms with (n_chosen, n_dimension) shape.
            tpye: Types of chosen atoms indexed from 0 with (n_chosen,) shape.
                Atoms with negative types are ignored.
            cell: The unit cell. For directions with no periodic boundary
                conditions the unit cell vetor has to be set to 0.

        """
        desc = descriptor_method(allpos, alltype, pos, type, cell)
        return desc.reshape((pos.shape[0], -1))

    return generate_descriptor


def create_generate_batch_descriptor(descriptor_method: Callable):
    """Creates a function to generate descriptors of a batch of structures.

    Args:
        descriptor_method: A function used to generate descriptors.

    Returns:
        A function which generates descriptors of the chosen atoms in a batch
        of structures. It uses five arguments, the position of all atoms, the
        types of all atoms, the positions of the chosen atoms, the types of the
        chosen atoms and the unit cell respectively.
    """

    def generate_descriptor(allpos, alltype, pos, type, cell):
        """Generates descriptors of chosen atoms in a single structure.

        Args:
            allpos: Positions of all atoms with (n_batch, n_atoms, n_dimension) shape.
            alltype: Types of all atoms indexed from 0 with (n_batch, n_atoms)
                shape. Atoms with negative types are ignored.
            pos: Positions of chosen atoms with (n_batch, n_chosen, n_dimension) shape.
            tpye: Types of chosen atoms indexed from 0 with (n_chosen, n_atoms).
                Atoms with negative types are ignored.
            cell: The batch of unit cells. For directions with no periodic
                boundary conditions the unit cell vetor has to be set to 0.

        """
        desc = descriptor_method(allpos, alltype, pos, type, cell)
        return desc.reshape((pos.shape[0], -1))

    return jax.vmap(generate_descriptor)


def create_generate_structures(
    generator: Generator, postprocess: Callable, n_latent: int
):
    """Creates a function to generate a batch of structures.

    Args:
        generator: The Generator object used to generate the input of the
            postprocessor.
        postprocess: A function which creates a structure based on the
            generator output. Has to return with positions of all atoms,
            types of all atoms, positions of chosen atoms,
            types of chosen atoms and the unit cell.
        n_latent: The number of latent variables.

    Returns:
        A function which generates n_batch structures given the generator
        weights and (n_batch, n_latent) latent variables. The generated
        structures consist of: the positions of all atoms, the types of all
        atoms, the positions of the chosen atoms, the types of the chosen atoms
        and the unit cell respectively.
    """

    def generate_single(params_gen, latent):
        intermediate = generator.apply(params_gen, latent)
        all_pos, all_type, pos, type, cell = postprocess(intermediate)

        return all_pos, all_type, pos, type, cell

    generate_batch = jax.jit(jax.vmap(generate_single, (None, 0), 0))

    def generate_structures(generator_params, key, n_struct):
        latent = jax.random.normal(key, shape=(n_struct, n_latent))
        return generate_batch(generator_params, latent)

    return generate_structures
