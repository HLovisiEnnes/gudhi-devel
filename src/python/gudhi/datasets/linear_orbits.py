"""
This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.

Author(s):       Henrique Ennes & Raphaël Tinarrage

Copyright (C) 2016 Inria & Institute of Science and Technology Austria
-----------------------------------------------------------------------------------------------------------------------


This module provides functions to sample points on orbits of representations, from their Lie algebra generators.
Namely, the Lie algebras are stored through bases (tuples of skew-symmetric matrices), that we suppose to be isomorphic
to the canonical pushforward algebras of the representations implemented in the module "algebra", through conjugation
by an orthogonal matrix. This assumption it crucial since, combined with the representation type of the algebra, we are
able to compute the periods of the basis elements (minimal t>0 such that exp(tA)=I), and hence be able to reconstruct
the orbit entirely (and without repetition for the torus). Note that, given the algebra alone, we cannot compute the
periods in a robust way.

-----------------------------------------------------------------------------------------------------------------------

Sample on orbits:
    sample_orbit_from_algebra
    sample_orbit_from_rep
    sample_orbit_from_group

-----------------------------------------------------------------------------------------------------------------------
"""

# Standard imports.
import math
from typing import Optional, List, Literal
import itertools

# Third-party imports.
import numpy as np
import scipy
from gudhi.liedetect.algebra import (
    get_random_lattice,
    get_random_constrained_partition,
    get_canonical_pushforward_algebra,
)
from gudhi.liedetect.linear_orbits_utils import(
    sample_orbit_from_algebra_su2,
    sample_orbit_from_algebra_torus
)


"""-
-----------------------------------------------------------------------------------------------------------------------
Sample on orbits
-----------------------------------------------------------------------------------------------------------------------
"""


def sample_orbit_from_algebra(
    group: str,
    rep_type: tuple,
    algebra: List[np.ndarray],
    x: np.ndarray,
    nb_points: int,
    method: Literal["uniform", "random"] = "uniform",
    verbose: bool = False,
) -> np.ndarray:
    """
    Samples points on the orbit of a compact Lie group representation, given its Lie algebra generators. We suppose
    that the algebra is isomorphic to the canonical algebra indicated in rep_type. This allows us to compute the
    periods, which are, otherwise, not stably computable from the algebra alone.

    Args:
        group (str): The group type, e.g., 'torus' or 'SU(2)'.
        rep_type (tuple): Representation type parameters (e.g., weights or partition).
        algebra (List[np.ndarray]): List of Lie algebra generators as matrices.
        x (np.ndarray): Initial vector to act on.
        nb_points (int): Number of points to sample.
        method (str): Sampling method, 'uniform' or 'random'. Defaults to 'uniform'.
        verbose (bool): Whether to print information about the sampled orbit.

    Returns:
        np.ndarray: Array of sampled points on the orbit.
    """
    if group == "torus":
        orbit = sample_orbit_from_algebra_torus(
            rep_type=rep_type, algebra=algebra, x=x, nb_points=nb_points, method=method
        )
    elif group in ["SU(2)", "SO(3)"]:
        orbit = sample_orbit_from_algebra_su2(
            rep_type=rep_type, algebra=algebra, x=x, nb_points=nb_points, method=method
        )
    else:
        raise ValueError(f"Group '{group}' not recognized.")
    if verbose:
        print(
            f"""Sampled {len(orbit)} {method} points on the orbit of
            \x1b[1;31m{group} with rep {rep_type}\x1b[0m."""
        )
    return orbit


def sample_orbit_from_rep(
    group: str,
    rep_type: tuple,
    nb_points: int,
    conjugate_algebra: bool = False,
    right_multiply_algebra: bool = False,
    translate_orbit: bool = False,
    method: Literal["uniform", "random"] = "uniform",
    verbose: bool = False,
) -> np.ndarray:
    """
    Samples points on the orbit of a compact Lie group representation, given its representation type. We suppose that
    the algebra is isomorphic to the canonical algebra indicated in rep_type. This allows us to compute the periods,
    which are, otherwise, not stably computable from the algebra alone.

    Note that the output orbit, in the uniform case, may not contain exactly nb_points points (this only happens if
    the parameter nb_points is a perfect power of the group dimension). In the random case, it will contain exactly
    nb_points points.

    Args:
        group (str): The group type, e.g., 'torus' or 'SU(2)'.
        rep_type (tuple): Representation type parameters (e.g., weights or partition).
        nb_points (int): Number of points to sample.
        conjugate_algebra (bool): Whether to conjugate the algebra by a random orthogonal matrix.
        right_multiply_algebra (bool): Whether to right-multiply the algebra by a random orthogonal matrix.
        translate_orbit (bool): Whether to translate the sampled orbit by a random orthogonal transformation.
        method (str): Sampling method, 'uniform' or 'random'. Defaults to 'uniform'.
        verbose (bool): Whether to print information about the sampled orbit.

    Returns:
        np.ndarray: Array of sampled points on the orbit.
    """
    ambient_dim = len(rep_type[0]) * 2 if group == "torus" else sum(rep_type)
    # Get canonical pushforward algebra.
    algebra = get_canonical_pushforward_algebra(group=group, rep_type=rep_type)
    # Get initial vector.
    x = np.ones(ambient_dim)
    x /= np.linalg.norm(x)
    # Conjugate algebra if needed.
    if conjugate_algebra:
        orth = scipy.stats.special_ortho_group.rvs(ambient_dim)
        algebra = [orth @ mat @ orth.T for mat in algebra]
        # Translate initial vector (to conserve a homogeneous orbit).
        x = orth @ x
    # Right-multiply algebra if needed.
    if right_multiply_algebra:
        if group == "torus":
            raise NotImplementedError("Right-multiply not implemented for torus representations.")
        orth = scipy.stats.special_ortho_group.rvs(3)
        algebra = [np.sum([algebra[j] * orth[j, i] for j in range(len(algebra))], axis=0) for i in range(len(algebra))]
    # Sample orbit.
    orbit = sample_orbit_from_algebra(
        group=group,
        rep_type=rep_type,
        algebra=algebra,
        x=x,
        nb_points=nb_points,
        method=method,
        verbose=verbose,
    )
    # Translate orbit if needed.
    if translate_orbit:
        orth = scipy.stats.special_ortho_group.rvs(ambient_dim)
        orbit = np.array([orth @ point for point in orbit])
    return orbit


def sample_orbit_from_group(
    group: str,
    ambient_dim: int,
    nb_points: int,
    frequency_max: Optional[int] = None,
    group_dim: Optional[int] = None,
    conjugate_algebra: bool = False,
    right_multiply_algebra: bool = False,
    translate_orbit: bool = False,
    method:  Literal["uniform", "random"] = "uniform",
    span_ambient_space: bool = False,
    verbose: bool = False,
) -> tuple[np.ndarray, tuple]:
    """
    Samples an orbit of a random representation in a given ambient space.

    Args:
        group (str): The group type, e.g., 'torus', 'SU(2)', or 'SO(3)'.
        ambient_dim (int): Dimension of the ambient space.
        nb_points (int): Number of points to sample on the orbit.
        frequency_max (Optional[int]): Maximal frequency for torus
            representations.
        group_dim (Optional[int]): Dimension of the group (for torus).
        conjugate_algebra (bool): Whether to conjugate the algebra by a
            random orthogonal matrix.
        right_multiply_algebra (bool): Whether to right-multiply the
            algebra by a random orthogonal matrix.
        translate_orbit (bool): Whether to translate the sampled orbit
            by a random orthogonal transformation.
        method (str): Sampling method, 'random' or 'uniform'.
        span_ambient_space (bool): Whether to only consider representations
            whose orbits span the ambient space. Only implemented for the
            circle or the non-Abelian groups.
        verbose (bool): Whether to print information about the sampled orbit.

    Returns:
        tuple[np.ndarray, tuple]: The sampled orbit (array of shape
            (nb_points, ambient_dim)) and representation type.
    """
    # Gets random representation
    if group == "torus":
        rep_type = get_random_lattice(
            lattice_rank=group_dim,
            ambient_rank=ambient_dim // 2,
            frequency_max=frequency_max,
            span_ambient_space=span_ambient_space,
        )

    elif group in ["SU(2)", "SO(3)"]:
        rep_type = get_random_constrained_partition(
            group=group, ambient_dim=ambient_dim, span_ambient_space=span_ambient_space
        )

    else:
        raise NotImplementedError(f"Group '{group}' not recognized.")

    # Samples orbit from the representation type
    orbit = sample_orbit_from_rep(
        group=group,
        rep_type=rep_type,
        nb_points=nb_points,
        conjugate_algebra=conjugate_algebra,
        right_multiply_algebra=right_multiply_algebra,
        translate_orbit=translate_orbit,
        method=method,
        verbose=verbose,
    )
    return orbit, rep_type
