""" This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
    Author(s):       Henrique Ennes & Raphaël Tinarrage

    Copyright (C) 2016 Inria
"""
import pytest
import numpy as np

from gudhi.liedetect import OrbitFitter, are_representations_equivalent
'''
Fixes data set.
'''


@pytest.fixture
def fix_pts():
    '''
    Builds orbit of weights (1,2) in R^4.
    '''
    thetas = np.linspace(0, 2*np.pi, 100)
    return np.array([(np.cos(theta), np.sin(theta),
                      np.cos(2 * theta), np.sin(2 * theta))
                    for theta in thetas])


'''
Test functions
'''


def test_lie_pca(fix_pts):
    orbit_fitter = OrbitFitter(fix_pts)
    nb_neighbors = 10
    orbit_dim = 1

    _ = orbit_fitter.lie_pca(nb_neighbors=nb_neighbors,
                             orbit_dim=orbit_dim,
                             method='PCA',
                             verbose=False)
    vals = orbit_fitter.print_lie_pca_eigenvalues(return_vals=True)
    assert abs(vals[0]) < 1e-2

    for i in vals[1:]:
        assert abs(i) > 1e-2

    orbit_fitter = OrbitFitter(fix_pts)
    nb_neighbors = 10
    orbit_dim = 1

    _ = orbit_fitter.lie_pca(nb_neighbors=nb_neighbors,
                             orbit_dim=orbit_dim,
                             method='covariance',
                             verbose=False)
    vals = orbit_fitter.print_lie_pca_eigenvalues(return_vals=True)
    assert abs(vals[0]) < 1e-2

    for i in vals[1:]:
        assert abs(i) > 1e-2


def test_project_lie_algebra(fix_pts):
    orbit_fitter = OrbitFitter(fix_pts)
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

    assert are_representations_equivalent('torus', la, ((1, 2),))

    method = "bottom_lie_pca"
    la, _ = orbit_fitter.closest_algebra(group='torus', group_dim=1,
                                         frequency_max=frequency_max,
                                         method=method, verbose=False)
    assert are_representations_equivalent('torus', la, ((1, 2),))

    method = "full_lie_pca"
    la, _ = orbit_fitter.closest_algebra(group='torus', group_dim=1,
                                         frequency_max=frequency_max,
                                         method=method, verbose=False)
    assert are_representations_equivalent('torus', la, ((1, 2),))


def test_sample_orbit(fix_pts):
    orbit_fitter = OrbitFitter(fix_pts)
    nb_neighbors = 10
    orbit_dim = 1

    _ = orbit_fitter.lie_pca(nb_neighbors=nb_neighbors,
                             orbit_dim=orbit_dim,
                             method='PCA',
                             verbose=False)

    frequency_max = 4
    method = "abelian"
    _, _ = orbit_fitter.closest_algebra(group='torus', group_dim=1,
                                        frequency_max=frequency_max,
                                        method=method, verbose=False)

    _ = orbit_fitter.sample_orbit(nb_points=1000)

    assert min(orbit_fitter.hausdorff_distances_) < 1e-2
