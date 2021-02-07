.. _api:

API
===

MDE
~~~

.. autoclass:: pymde.MDE
    :members:
    :exclude-members: forward

    .. automethod:: __init__

.. autofunction:: pymde.preserve_neighbors

.. autofunction:: pymde.preserve_distances

Distortion functions
~~~~~~~~~~~~~~~~~~~~

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

Constraints
~~~~~~~~~~~

.. autoclass:: pymde.constraints.Constraint
   :members:

.. autofunction:: pymde.Centered

.. autofunction:: pymde.Standardized

.. autoclass:: pymde.Anchored
   
   .. automethod:: __init__


Preprocessing
~~~~~~~~~~~~~

.. autoclass:: pymde.Graph
    :members:

.. autofunction:: pymde.preprocess.graph.k_nearest_neighbors

.. autofunction:: pymde.preprocess.graph.shortest_paths

.. autofunction:: pymde.preprocess.data_matrix.k_nearest_neighbors

Utilities
~~~~~~~~~

.. autofunction:: pymde.all_edges

.. autofunction:: pymde.align

.. autofunction:: pymde.center

.. autofunction:: pymde.rotate

.. autofunction:: pymde.plot
