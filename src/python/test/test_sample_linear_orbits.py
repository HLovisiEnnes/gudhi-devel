""" This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
    Author(s):       Henrique Ennes & Raphaël Tinarrage

    Copyright (C) 2016 Inria


    Obs: More robust tests will be implemented once the full LieDetect pipeline is ready.
"""
import numpy as np

from gudhi.datasets.linear_orbits import (
    sample_from_lie_algebra,
    sample_from_lie_group_rep,
    sample_from_lie_group
)
import gudhi.liedetect as liedetect

def test_sampling():
    group = 'torus'
    ambient_dim = 4
    nb_points = 100
    frequency_max = 2
    group_dim = 1
    method = 'uniform'
    span_ambient_space = True
    rep_type = ((1, 2),)
    algebra = [
        np.array([[0, 1, 0, 0],
                  [-1, 0, 0, 0],
                  [0, 0, 0, 2],
                  [0, 0, -2, 0]])
    ]
    x = np.array([1, 0, 1, 0])


    pts_algebra = sample_from_lie_algebra(
        group,
        rep_type,
        algebra,
        x,
        nb_points,
        method=method
    )

    pts_rep = sample_from_lie_group_rep(group,
                                rep_type,
                                nb_points,
                                method=method)

    assert np.allclose(pts_algebra, pts_rep, atol=1e-6)


    pts_group, _ = sample_from_lie_group(
        group,
        ambient_dim,
        nb_points,
        frequency_max,
        group_dim,
        method=method,
        span_ambient_space=span_ambient_space
    )


    assert pts_group.shape[1] == ambient_dim
