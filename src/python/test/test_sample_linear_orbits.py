""" This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
    Author(s):       Henrique Ennes & Raphaël Tinarrage

    Copyright (C) 2016 Inria
"""
import numpy as np

from gudhi.datasets.linear_orbits import (
    sample_orbit_from_algebra,
    sample_orbit_from_rep,
    sample_orbit_from_group
)
import gudhi.liedetect as liedetect

def test_sample_from_algebra():
    group = 'torus'
    rep_type = ((1, 2),)
    algebra = [
        np.array([[0, 1, 0, 0],
                  [-1, 0, 0, 0],
                  [0, 0, 0, 2],
                  [0, 0, -2, 0]])
    ]
    x = np.array([1, 0, 1, 0])
    nb_points = 100
    method = 'uniform'

    pts = sample_orbit_from_algebra(
        group,
        rep_type,
        algebra,
        x,
        nb_points,
        method=method
    )

    orbit_fitter = liedetect.OrbitFitter(pts)
    nb_neighbors = 10
    orbit_dim = 1

    _ = orbit_fitter.lie_pca(nb_neighbors=nb_neighbors,
                             orbit_dim=orbit_dim,
                             method='PCA',
                             verbose=False)

    frequency_max = 4

    method = "abelian"
    la, _ = orbit_fitter.closest_algebra(group='torus', group_dim=1,
                                         frequency_max=frequency_max,
                                         method=method, verbose=False)

    assert liedetect.are_representations_equivalent('torus', rep_type, la)


def test_sample_from_rep():
    group = 'torus'
    rep_type = ((1, 2),)
    nb_points = 100
    method = 'uniform'

    pts = sample_orbit_from_rep(group,
                                rep_type,
                                nb_points,
                                method=method)

    orbit_fitter = liedetect.OrbitFitter(pts)
    nb_neighbors = 10
    orbit_dim = 1

    _ = orbit_fitter.lie_pca(nb_neighbors=nb_neighbors,
                             orbit_dim=orbit_dim,
                             method='PCA',
                             verbose=False)

    frequency_max = 4

    method = "abelian"
    la, _ = orbit_fitter.closest_algebra(group='torus', group_dim=1,
                                         frequency_max=frequency_max,
                                         method=method, verbose=False)

    assert liedetect.are_representations_equivalent('torus', rep_type, la)


def test_sample_from_group():
    group = 'torus'
    ambient_dim = 4
    nb_points = 100
    frequency_max = 2
    group_dim = 1
    method = 'uniform'
    span_ambient_space = True

    pts, rep_type = sample_orbit_from_group(
        group,
        ambient_dim,
        nb_points,
        frequency_max,
        group_dim,
        method=method,
        span_ambient_space=span_ambient_space
    )

    orbit_fitter = liedetect.OrbitFitter(pts)
    nb_neighbors = 10
    orbit_dim = 1

    _ = orbit_fitter.lie_pca(nb_neighbors=nb_neighbors,
                             orbit_dim=orbit_dim,
                             method='PCA',
                             verbose=False)

    frequency_max = 4

    method = "abelian"
    la, _ = orbit_fitter.closest_algebra(group='torus', group_dim=1,
                                         frequency_max=frequency_max,
                                         method=method, verbose=False)

    assert liedetect.are_representations_equivalent('torus', rep_type, la)
