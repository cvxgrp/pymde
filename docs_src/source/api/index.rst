.. _api:

API documentation
=================

.. _api_mde:

.. note:: 

   PyMDE is very young software. When updates are released, we will try to be
   backward compatible with earlier versions, but sometimes we may be unable to
   do so.

   If a version ever includes a breaking change, we will make sure
   to communicate that clearly. Until we reach v1.0.0, an increase in the minor
   version will indicate breaking changes (e.g., v0.2.0 may have some changes
   that are incompatible with v0.1.0, but v0.1.5 will be fully compatible
   with v0.1.0).


MDE
---

.. autoclass:: pymde.MDE
    :members:
    :exclude-members: forward

    .. automethod:: __init__

.. autoclass:: pymde.optim.SolveStats

.. _api_preserve_neighbors:

Preserve neighbors
------------------

.. autofunction:: pymde.preserve_neighbors

.. _api_preserve_distances:

Preserve distances
------------------

.. autofunction:: pymde.preserve_distances

Distortion functions
--------------------

Penalties
^^^^^^^^^

.. automodule:: pymde.penalties

   .. autoclass:: pymde.penalties.Linear
        :exclude-members: forward

   .. autoclass:: pymde.penalties.Quadratic
        :exclude-members: forward

   .. autoclass:: pymde.penalties.Cubic
        :exclude-members: forward

   .. autoclass:: pymde.penalties.Power
        :exclude-members: forward

   .. autoclass:: pymde.penalties.Huber
        :exclude-members: forward

   .. autoclass:: pymde.penalties.Logistic
        :exclude-members: forward

   .. autoclass:: pymde.penalties.Log1p
        :exclude-members: forward

   .. autoclass:: pymde.penalties.Log
        :exclude-members: forward

   .. autoclass:: pymde.penalties.InvPower
        :exclude-members: forward

   .. autoclass:: pymde.penalties.LogRatio
        :exclude-members: forward

   .. autoclass:: pymde.penalties.PushAndPull
        :exclude-members: forward

Losses
^^^^^^^^^

.. automodule:: pymde.losses

   .. autoclass:: pymde.losses.Absolute
        :exclude-members: forward

   .. autoclass:: pymde.losses.Quadratic
        :exclude-members: forward

   .. autoclass:: pymde.losses.Cubic
        :exclude-members: forward

   .. autoclass:: pymde.losses.Power
        :exclude-members: forward

   .. autoclass:: pymde.losses.WeightedQuadratic
        :exclude-members: forward

   .. autoclass:: pymde.losses.Fractional
        :exclude-members: forward

   .. autoclass:: pymde.losses.SoftFractional
        :exclude-members: forward

.. _api_constraints:

Constraints
-----------

.. autoclass:: pymde.constraints.Constraint
   :members:

.. autofunction:: pymde.Centered

.. autofunction:: pymde.Standardized

.. autoclass:: pymde.Anchored
   
   .. automethod:: __init__

Preprocessing
-------------

.. _api_graph:

.. autoclass:: pymde.Graph
    :members:

.. autofunction:: pymde.preprocess.data_matrix.k_nearest_neighbors

.. autofunction:: pymde.preprocess.graph.k_nearest_neighbors

.. autofunction:: pymde.preprocess.dissimilar_edges

.. autofunction:: pymde.preprocess.graph.shortest_paths


Classical embeddings
--------------------

.. autofunction:: pymde.quadratic.pca

.. autofunction:: pymde.quadratic.spectral

Utilities
---------

.. autofunction:: pymde.all_edges

.. autofunction:: pymde.align

.. autofunction:: pymde.center

.. autofunction:: pymde.rotate

.. autofunction:: pymde.plot

Datasets
--------

.. autoclass:: pymde.datasets.Dataset
   :members:

.. autofunction:: pymde.datasets.MNIST

.. autofunction:: pymde.datasets.google_scholar

.. autofunction:: pymde.datasets.covid19_scrna_wilk
