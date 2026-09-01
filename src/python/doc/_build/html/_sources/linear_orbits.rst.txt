
:orphan:

.. To get rid of WARNING: document isn't included in any toctree

=============
Linear orbits
=============

The module **linear_orbits** generates points in orbits of representations of :math:`SO(2)`, :math:`T^{d}`, :math:`SO(3)`,
and :math:`SU(2)` in real, finite-dimensional vector spaces.


Definitions
-----------
A Lie group :math:`G` corresponds to a group with a smooth manifold structure, compatible with the group's product. Here, we will
assume that :math:`G` is either :math:`SO(2)`, :math:`T^{d}`, for :math:`d>0`, :math:`SO(3)`, or :math:`SU(2)`. Notice that the 
group :math:`T^{1}` is actually equal to :math:`SO(2)`, and, for this reason, we shall call it a torus group as well.

A representation or linear action is a smooth group morphism :math:`\rho: G \to GL(\mathbb{R}, n)`, where :math:`GL(\mathbb{R}, n)` corresponds to the
group of :math:`n \times n` invertible real matrices.

Given a non-zero point :math:`x\in \mathbb{R}^{n}`, the orbit of :math:`x` under :math:`\rho` is :math:`O_x=\{\rho(g)\cdot x| g \in G\}`.
Orbits of :math:`T^d` are dimension :math:`d` submanifolds of :math:`\mathbb{R}^{n}`, whereas orbits of :math:`SO(3)`, and :math:`SU(2)`
are both of dimension :math:`3`. The **linear_orbits** generates a point cloud close to or exactly in an orbit of the families 
of Lie groups considered.


Uses
----
Three main functions are provided to generate the orbits. The first method allows for sampling from a randomly chosen type
representation orbit of the specified Lie group.

.. code-block:: python

   from gudhi.datasets.linear_orbits import sample_from_lie_group

   # Generates 500 uniformly spaces points from a SO(2) representation orbit in R^4
   # This is equivalent to a linearly embedded torus knot in R^4
   gen_points = sample_from_lie_group(group = "torus", ambient_dim = 4, nb_points = 500)

.. autofunction:: gudhi.datasets.linear_orbits.sample_from_lie_group


The representation type can be explicitly given.

.. code-block:: python

   from gudhi.datasets.linear_orbits import sample_from_lie_group_rep

    # Generates 500 uniformly spaces points from the orbit of a SO(2) representation of type (1,2)
    gen_points = sample_from_lie_group_rep(group = "torus", rep_type = ((1, 2),), nb_points = 500)

.. autofunction:: gudhi.datasets.linear_orbits.sample_from_lie_group_rep

The last method allows for sampling from a specific representation orbit of the specified Lie group.

.. code-block:: python

   from gudhi.datasets.linear_orbits import sample_from_lie_algebra
    alg = [
        np.array([[0, 1, 0, 0],
                  [-1, 0, 0, 0],
                  [0, 0, 0, 2],
                  [0, 0, -2, 0]])
    ]
    # Generates 500 uniformly spaces points from the orbit of a SO(2) representation of type (1,2) with given algebra
   gen_points = sample_from_lie_algebra(group = "torus", rep_type =  = ((1, 2),), algebra = alg, nb_points = 500)

.. autofunction:: gudhi.datasets.linear_orbits.sample_from_lie_algebra




