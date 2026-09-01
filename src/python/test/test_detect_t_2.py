""" This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
    Author(s):       Henrique Ennes & Raphaël Tinarrage

    Copyright (C) 2016 Inria
"""
import pytest
import numpy as np

from gudhi.liedetect import OrbitFitter
'''
Fixes data set.
'''


@pytest.fixture
def fix_pts():
    '''
    Builds orbit of weights 1 in R^4.
    '''
    thetas1 = np.linspace(0, 2*np.pi, 100)
    thetas2 = np.linspace(0, 2*np.pi, 100)
    return np.array([(np.cos(theta1), np.sin(theta1),
                      np.cos(theta2), np.sin(theta2))
                    for theta1 in thetas1
                    for theta2 in thetas2])


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
    assert abs(vals[0]) < 0.15
    assert abs(vals[1]) < 0.15

    for i in vals[2:]:
        assert abs(i) > 0.15

    orbit_fitter = OrbitFitter(fix_pts)
    nb_neighbors = 10
    orbit_dim = 1

    _ = orbit_fitter.lie_pca(nb_neighbors=nb_neighbors,
                             orbit_dim=orbit_dim,
                             method='covariance',
                             verbose=False)
    vals = orbit_fitter.print_lie_pca_eigenvalues(return_vals=True)
    assert abs(vals[0]) < 0.15
    assert abs(vals[1]) < 0.15

    for i in vals[2:]:
        assert abs(i) > 0.15


def test_project_lie_algebra(fix_pts):
    orbit_fitter = OrbitFitter(fix_pts)
    nb_neighbors = 30
    orbit_dim = 2

    _ = orbit_fitter.lie_pca(nb_neighbors=nb_neighbors,
                             orbit_dim=orbit_dim,
                             method='PCA',
                             verbose=False)

    frequency_max = 4

    method = "abelian"
    la, _ = orbit_fitter.closest_algebra(group='torus', group_dim=2,
                                         frequency_max=frequency_max,
                                         method=method, verbose=False)

    print(la)
    assert abs(sum(la[0])) == 1
    assert abs(sum(la[1])) == 1

    method = "bottom_lie_pca"
    la, _ = orbit_fitter.closest_algebra(group='torus', group_dim=2,
                                         frequency_max=frequency_max,
                                         method=method, verbose=False)
    assert abs(sum(la[0])) == 1
    assert abs(sum(la[1])) == 1

    method = "full_lie_pca"
    la, _ = orbit_fitter.closest_algebra(group='torus', group_dim=2,
                                         frequency_max=frequency_max,
                                         method=method, verbose=False)
    assert abs(sum(la[0])) == 1
    assert abs(sum(la[1])) == 1


def test_sample_orbit(fix_pts):
    orbit_fitter = OrbitFitter(fix_pts)
    nb_neighbors = 30
    orbit_dim = 2

    _ = orbit_fitter.lie_pca(nb_neighbors=nb_neighbors,
                             orbit_dim=orbit_dim,
                             method='PCA',
                             verbose=False)

    frequency_max = 4
    method = "abelian"
    _, _ = orbit_fitter.closest_algebra(group='torus', group_dim=2,
                                        frequency_max=frequency_max,
                                        method=method, verbose=False)

    _ = orbit_fitter.sample_orbit(nb_points=5000)

    assert min(orbit_fitter.hausdorff_distances_) < 0.15
