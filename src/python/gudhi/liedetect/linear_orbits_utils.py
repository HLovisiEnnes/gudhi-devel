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
    get_periods_torus
    sample_orbit_from_algebra_su2
    sample_orbit_from_algebra_torus
-----------------------------------------------------------------------------------------------------------------------
"""

# Standard imports.
import itertools
import math
from typing import Literal

# Third-party imports.
import numpy as np
import scipy

import gudhi.subsampling

"""-
-----------------------------------------------------------------------------------------------------------------------
Sample on orbits
-----------------------------------------------------------------------------------------------------------------------
"""


def get_periods_torus(
    rep_type: tuple,
    algebra: list[np.ndarray],
) -> list[float]:
    """
    Returns a list of minimal periods t>0 such that exp(tA)=I for each A in the pushforward algebra. We suppose that
    the elements in "algebra" are such that they generate a periodic 1-parameter subgroup. More precisely, we suppose
    that the bijection between "algebra" and the canonical algebra (implemented in get_canonical_pushforward_algebra)
    is a Lie algebra isomorphism, induced by a conjugation. In particular, up to normalization by the norm of the
    elements, the periods are identical.

    Case of SO(2):
        The period of the integer matrix A* representing the pushforward algebra of SO(2) via the rep (a1, ..., am) is
                2 * pi / gcd(a1, ..., am).
        Its norm is
                sqrt(2) * ||(a1, ..., am)||.
        In particular, if A is a skew-symmetric matrix that is, up to normalization, conjugate to A*, then its period
            is
                2 * pi / gcd(a1, ..., am) * 2 * ||(a1, ..., am)|| / ||A||.

    Case of T^d:
        Similar to the case of SO(2), but reasoning coordinate by coordinate.
    """

    def period_so2(weights: tuple[int, ...]) -> float:
        return 2 * np.pi / math.gcd(*weights)

    def norm_so2(weights: tuple[int, ...]) -> float:
        return np.sqrt(2) * np.linalg.norm(weights)

    # Compute the periods
    periods = [
        period_so2(weights) * norm_so2(weights) / np.linalg.norm(mat)
        for weights, mat in zip(rep_type, algebra)
    ]

    # Sanity check: the exponentiated matrices should be the identity
    for period, mat in zip(periods, algebra):
        if not np.isclose(scipy.linalg.expm(period * mat), np.eye(len(mat))).all():
            print(
                "Error! Incorrect period. Distance to identity:",
                np.linalg.norm(scipy.linalg.expm(period * mat) - np.eye(len(mat))),
            )
    return periods


def sample_orbit_from_algebra_torus(
    rep_type: tuple,
    algebra: list[np.ndarray],
    x: np.ndarray,
    nb_points: int,
    method: Literal["uniform", "random"] = "uniform",
) -> np.ndarray:
    """
    Sample points on the orbit of a torus representation from its Lie algebra. We suppose that the algebra is
    the canonical algebra indicated in rep_type.

    Args:
    rep_type (tuple): Representation type parameters (e.g., weights).
    algebra (list[np.ndarray]): List of Lie algebra generators as matrices.
    x (np.ndarray): Initial vector to act on.
    nb_points (int): Number of points to sample.
    method (str): Sampling method, 'uniform' or 'random'. Defaults to 'uniform'.

    Returns:
        np.ndarray: Array of sampled points on the orbit.
    """
    # Gets periods
    periods = np.asarray(get_periods_torus(rep_type, algebra), dtype=float)
    group_dim = len(algebra)

    # Generates grid based on the method.
    if method == "random":
        # Generates random points in hypercube
        # [0, period[0]] x ... x [0, period[-1]]
        times = np.random.rand(nb_points, group_dim) * periods
    elif method == "uniform":
        # Get number of points (potentially too many, sparsify later).
        nb_points_circle = int(np.ceil(nb_points ** (1.0 / group_dim)))
        grids = [
            np.linspace(0.0, periods[i], nb_points_circle, endpoint=False)
            for i in range(group_dim)
        ]
        times = np.array(list(itertools.product(*grids)), dtype=float)
    else:
        raise ValueError(
            f"Method '{method}' not recognized. Use 'uniform' or 'random'."
        )

    # Generates orbit (linear combinations or algebra elements wrt times)
    orbit = np.empty((len(times), x.size), dtype=float)
    for k, t_vec in enumerate(times):
        mat_alg = np.zeros_like(algebra[0], dtype=float)
        for t, mat in zip(t_vec, algebra):
            mat_alg += t * mat
        orbit[k] = scipy.linalg.expm(mat_alg) @ x
    # Sparsify if required.
    if len(orbit) > nb_points:
        orbit = gudhi.subsampling.choose_n_farthest_points(
            points=orbit, nb_points=nb_points
        )
    return orbit


def sample_orbit_from_algebra_su2(
    rep_type: tuple,
    algebra: list[np.ndarray],
    x: np.ndarray,
    nb_points: int,
) -> np.ndarray:
    """
    Sample via Haar (uniform) measure. Euler angle factorization:
            g(alpha, b, c) = exp(alpha Az) x exp(beta Ay) x exp(gamma Az)
    where
        alpha ~ Uniform[0, 2π)
        beta ~ arccos(Uniform[-1, 1])
        gamma ~ Uniform[0, 4π) for SU(2) and Uniform[0, 2π) for SO(3)

    Note: "uniform" not implemented, behaves as "random".

    Args:
        rep_type (tuple): Representation type parameters (e.g., weights or partition).
        algebra (list[np.ndarray]): List of Lie algebra generators as matrices.
        x (np.ndarray): Initial vector to act on.
        nb_points (int): Number of points to sample.

    Returns:
        np.ndarray: Array of sampled points on the orbit.
    """

    def period_irrep_su2(dim: int) -> float:
        if dim % 4 == 0:
            return 4 * np.pi
        else:
            return 2 * np.pi

    # Gets algebra basis
    algebra_y = algebra[1]
    algebra_z = algebra[2]

    # Random sample
    # alpha ~ Uniform[0, 2π)
    alpha = np.random.rand(nb_points) * (2 * np.pi)
    # beta ~ arccos(Uniform[-1, 1])
    beta = np.arccos(2 * np.random.rand(nb_points) - 1.0)
    # gamma ~ Uniform[0, 4π) or Uniform[0, 2π)
    period = max(period_irrep_su2(dim) for dim in rep_type)
    gamma = np.random.rand(nb_points) * period

    orbit = np.empty((nb_points, x.size), dtype=float)
    for i in range(nb_points):
        g = (
            scipy.linalg.expm(alpha[i] * algebra_z)
            @ scipy.linalg.expm(beta[i] * algebra_y)
            @ scipy.linalg.expm(gamma[i] * algebra_z)
        )
        orbit[i] = g @ x
    return orbit
