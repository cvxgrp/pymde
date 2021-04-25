.. _mde:


MDE
===

The central concept in PyMDE is the Minimum-Distortion Embedding (MDE) problem.
An MDE problem is an optimization problem whose solution is an embedding.

We can interpret an MDE problem as a declarative description of the properties
an embedding should satisfy. MDE problems in PyMDE are represented by
the :any:`pymde.MDE` object.

In this part of the tutorial, we explain what an MDE problem is. Then we
show how to construct MDE problems using PyMDE, with custom objective
functions and constraints. We also describe some of the useful methods that
the :any:`pymde.MDE` class provides, and how to sanity-check and compare
embeddings.


The MDE framework
-----------------

In this section, we introduce the concept of an MDE problem, whose solution
is an embedding. At a high-level, the objective of an MDE problem
is to minimally distort known relationships between some pairs of items,
while possibly satisfying some constraints.

We first explain this abstractly. In the next section, we show how to make MDE
problems using PyMDE.

Embedding matrix
^^^^^^^^^^^^^^^^

An MDE problem starts with a set :math:`\mathcal{V}` of :math:`n` items,
:math:`\mathcal{V} = \{0, 1, ..., n - 1\}`. An embedding of the set of items is
a matrix :math:`X \in \mathbf{R}^{n \times m}`,
where :math:`m` is the *embedding dimension*. The rows of :math:`X` are denoted
:math:`x_0, x_1, \ldots, x_{n-1}`, with :math:`x_i` being the *embedding vector*
associated with item :math:`i`. The quality of an embedding will depend only on
the Euclidean distances between the embedding vectors,

.. math::

  d_{ij} = \|x_i - x_j\|_2, \quad i, j = 0, \ldots, n - 1.

Distortion functions
^^^^^^^^^^^^^^^^^^^^
We make our preferences on the
:math:`d_{ij}` concrete with *distortion functions* associated with the
edges. These have the form

.. math::

  f_{ij} : \mathbf{R}_{+} \to \mathbf{R}_+, \quad (i, j) \in \mathcal E

where :math:`\mathcal E` is a set of edges :math:`(i , j)`, with
:math:`0 \leq i < j < n`. This set of edges may contain all pairs, or a subset
of pairs, but it must be non-empty.

For convenience, we can assume the edges are listed in some fixed order,
and label them as `1, 2, \ldots, p`, where :math:`p = |\mathcal E|`.
We can then represent the collection of distortion functions as a single
vector distortion function :math:`f : \mathbf{R}^p \to \mathbf{R}^p`, where
:math:`p = |\mathcal E|` and :math:`f_k` is the distortion function associated
with the :math:`k`-th edge.

Distortion functions are usually derived either from weights, or from
original distances or deviations between some pairs of items. PyMDE
provides a library of both types of distortion functions, in
:any:`pymde.penalties` and :any:`pymde.losses`.

Distortion functions from weights
"""""""""""""""""""""""""""""""""
We start with nonzero weights :math:`w_1, w_2, \ldots, w_p`, one for each edge.
A positive weight means the items in the edge are similar, and a negative
weight means they are disimilar. The larger the weight is, the more similar
the items are; the more negative the weight, the more dissimilar.

A vector distortion function  :math:`f : \mathbf{R}^{p} \to \mathbf{R}^p`
derived from weights has component functions

.. math::

    f_k(d) = \begin{cases}
        w_k p_{\text{attractive}}(d_k) & w_k > 0 \\
        w_k p_{\text{repulsive}}(d_k) & w_k < 0 \\
    \end{cases}.

where :math:`w_k` is a scalar weight and :math:`p_{\text{attractive}}` and
:math:`p_{\text{repulsive}}` are penalty functions. Penalty functions are
increasing functions: the attractive penalty encourages the distances to
be small, while the repulsive penalty encourages them to be large, or at least.
not small.

Attractive penalties are 0 when the input is 0, and grow otherwise. The
attractive and repulsive penalties can be the same, e.g. they can both be
quadratics :math:`d \mapsto d^2`, or they can be different. Typically, though,
repulsive penalties go to negative infinity as the input approaches 0, and
to 0 as the input grows large.

Distortion functions from deviations
""""""""""""""""""""""""""""""""""""
A vector distortion function :math:`f : \mathbf{R}^{p} \to \mathbf{R}^p`
derived from original deviations has component functions

.. math::

    f_k(d_k) = \ell(d_k, \delta_k), \quad k=1, \ldots, p,

where
:math:`\ell` is a loss function,
and :math:`\delta_k` is a scalar deviation or dissimilarity score associated
with the :math:`k`-th edge.

The deviations can be interpreted as targets for the embedding distances:
the loss function is 0 when :math:`d_k = \delta_k`, and positive otherwise.
So a deviation :math:`\delta_k`` of 0 means that the items in the k-th edge
are the same, and the larger the deviation, the more dissimilar the items are.

The simplest example of a loss function is the squared loss

.. math::

  \ell(d, \delta) = (d - \delta)^2.

Average distortion
^^^^^^^^^^^^^^^^^^

The value :math:`f_{ij}(d_{ij})` is the *distortion* associated for the
pair :math:`(i, j) \in \mathcal E`. The smaller the distortion, the better
the embedding captures the relationship between :math:`i` and :math:`j`.

The goal is to minimize the *average distortion* of the embedding `X`, defined
as

.. math::

  E(X) = \frac{1}{|\mathcal E|} \sum_{(i, j) \in \mathcal E} f_{ij} (d_{ij}),

possibly subject to the constraint that :math:`X \in \mathcal X`, where
:math:`\mathcal X` is a set of permissible embeddings. This gives the
optimization problem

.. math::

  \begin{array}{ll}
  \mbox{minimize} & E(X) \\
  \mbox{subject to} & X \in \mathcal X.
  \end{array}

This optimization problem is called an **MDE problem**. Its solution
is the embedding.

Constraints
^^^^^^^^^^^
We can optionally impose constraints on the embedding.

For example, we can enforce the embedding vectors to be **standardized**,
which means that they are centered and identity covariance, that is,
:math:`(1/n) X^T X = I`. When a standardization constraint is imposed,
the embedding problem always has a solution. Additionally, the standardization
constraint forces the embedding to spread out. When using distortion functions
from weights, this means we do not need repulsive penalties (but can choose
to include them anyway).

Or, we can **anchor** or pin some of the embedding vectors to fixed values.

Constructing an MDE problem
---------------------------

In PyMDE, instances of the :any:`pymde.MDE` class are MDE problems. The
:any:`pymde.preserve_neighbors` and :any:`pymde.preserve_distances`
functions we saw in the previous part of the tutorial both returned
``MDE`` instances.

To create an MDE instance, we need to specify five things:

* the number of items;
* the embedding dimension;
* the list of edges (a ``torch.Tensor``, of shape ``(n_edges, 2)``)
* the vector distortion function; and
* an optional constraint.

Let's walk through a very simple example.

Items
^^^^^
Let's say we have five items.
In PyMDE, items are represented by consecutive integer labels, in our case
0, 1, 2, 3, and 4.

Edges
^^^^^
Say we know that item 0 is similar to items 1 and 4, 1 is
similar to 2, 2 is similar to 3, and 3 is similar to 4. We include these
pairs in a list of edges

.. code:: python3

  edges = torch.tensor([[0, 1],  [0, 4], [1, 2], [2, 3]])

Distortion function
^^^^^^^^^^^^^^^^^^^
Next, we need to encode the fact that each edge
represents some degree of similarity between the items it contains. We'll use a
quadratic penalty :math:`f_k(d_k) = w_k d_k^2` (other choices are possible).
We'll associate a weight 1 to the first edge, 2 to the second edge, 5 to the
third edge, and 6 to the fourth edge; this conveys that 0 is somewhat similar
to 1 but more similar to 4, 2 is yet more similar to 3, and 3 is yet more
similar to 4. We write this in PyMDE as

.. code:: python3

  weights = torch.tensor([1., 2., 5., 6.])
  f = pymde.penalties.Quadratic(weights)

Constraint
^^^^^^^^^^
The last thing to specify is the constraint. Since we're using a distortion
function based on only positive weights, we'll need a standardization
constraint.

.. code:: python3

   constraint = pymde.Standardized()

Construction
^^^^^^^^^^^^
We can now construct the MDE problem:

.. code:: python3

  import pymde

  mde = pymde.MDE(
    n_items=5,
    embedding_dim=2,
    edges=edges,
    distortion_function=f,
    constraint=pymde.Standardized())

The ``mde`` object represents the MDE problem whose goal is to minimize
the average distortion with respect to ``f``, subject to the standardization
constraint. This object can be thought of describing the kind of embedding
we would like.

Embedding
^^^^^^^^^
To obtain the embedding, we call the :any:`pymde.MDE.embed` method:

.. code:: python3

   embedding = mde.embed()
   print(embedding)

.. code:: python3

   tensor([[ 0.0894, -1.8689],
           [-0.7726, -0.1450],
           [-0.6687,  0.5428],
           [-0.5557,  0.9696],
           [ 1.9077,  0.5015]])

We can check that the embedding is standardized with the following code:

.. code:: python3

  print(embedding.mean(axis=0))
  print((1/mde.n_items)*embedding.T @ embedding)

.. code:: python3

   tensor([4.7684e-08, 5.9605e-08])
   tensor([[1.0000e+00, 7.4506e-08],
           [7.4506e-08, 1.0000e+00]])

We can also evaluate the average distortion:

.. code:: python3

   print(mde.average_distortion(embedding))

.. code:: python3

   tensor(6.2884)

Summary
^^^^^^^
This very simple example showed all the components required
to construct an MDE problem. The full documentation for the MDE
class is available in the :any:`API documentation <api_mde>`.

In the next section, we'll learn more about distortion functions and
how to create them.

Distortion functions
--------------------

A distortion function is just a Python callable that maps the embedding
distances to distortions, using PyTorch operations. Its call signature should
be

.. code:: python3

   torch.Tensor(shape=(n_edges,), dtype=torch.float) -> torch.Tensor(shape=(n_edges,), dtype=torch.float)

For example, the quadratic penalty we used previously can be implemented as

.. code:: python3

  weights = torch.tensor([1., 2., 5., 6.]
  def f(distances):
    return weights * distances.pow(2)

A quadratic penalty based on original deviations could be implemented as

.. code:: python3

  deviations = torch.tensor([1., 2., 5., 6.]
  def f(distances):
    return (distances - deviations).pow(2)

In many applications, you won't need to implement your own distortion functions.
Instead, you can choose one from a library of useful distortion functions
that PyMDE provides.

PyMDE provides two types of distortion functions: penalties, which are 
based on weights, and losses, based on original deviations. (A natural
question is: Where do the weights or original deviations come from?
We'll see some recipes for creating edges and their weights / deviations in the
next part of the tutorial, which covers :any:`preprocessing <preprocess>`.)

Penalties
^^^^^^^^^

.. automodule:: pymde.penalties
   :noindex:

Losses
^^^^^^

.. automodule:: pymde.losses
   :noindex:


Constraints
-----------

PyMDE currently provides three constraint sets:

- :any:`pymde.Centered`, which constrains the embedding vectors to have mean zero;
- :any:`pymde.Standardized`, which constrains the embedding vectors to have identity covariance (and have mean zero);
- :any:`pymde.Anchored`, which pins specific items (called anchors) to specific values (i.e., this is an equality constraint on a subset of the embedding vectors).

Centered
^^^^^^^^^
If a constraint is not specified, the embedding will be centered, but no other
restrictions will be placed on it. Centering is without loss of generality,
since translating all the points does not affect the average distortion.

To explicitly create a centering constraint, use

.. code:: python3

   constraint = pymde.Centered()

Standardized
^^^^^^^^^^^^^^^

The standardization constraint is

.. math::

   (1/n) X^T X = I, \quad X^T \mathbf{1} = 0

where :math:`n` is the number of items (i.e., the number of rows in :math:`X`)
and :math:`\mathbf{1}` is the all-ones vector.

A standardization constraint can be created with

.. code:: python3

   constraint = pymde.Standardized()

The standardization constraint has several implications.

- It forces the embedding to spread out.
- It constrains sum of embedding distances to have a root-mean-square
  value of :math:`\sqrt{(2nm)/(n-1)}`, where :math:`m` is the embedding
  dimension. We call this value the natural length of the embedding.
- It makes the columns of the embedding uncorrelated, which can be useful if
  the embedding is to be used as features in a supervised learning task.

When the distortion function is based on penalties and all the weights
are positive, you **must** impose a standardization constraint, which will
force the embedding to spread out. When the weights are not all positive,
a standardization constraint is not required, but is recommended: MDE
problems with standardization constraints always have a solution. Without
the constraint, problems can sometimes be pathological.

When the distortion functions are based on losses, care must be taken to ensure
that the original deviations and embedding distances are on the same scale.
This can be done by rescaling the original deviations to have RMS equal
to the natural length.

Anchored
^^^^^^^^

The anchor constraint is

.. math::

   x_i = v_i, \quad i \in \text{anchors}

where :math:`\text{anchors}` is a subset of the items and :math:`v_i`
is a concrete value to which :math:`x_i` should be pinned.

An anchor constraint can be created with

.. code:: python3

   # anchors holds the item numbers that should be pinned
   anchors = torch.tensor([0., 1., 3.])
   # the ith row of values is the value v_i for the ith item in anchors
   values = torch.tensor([
     [0., 0.],
     [1., 2.],
     [-1., -1.],
   ])
   constraint = pymde.Anchored(anchors, values)


Below is a GIF showing the creation of an embedding of a binary tree,
in which the leaves have been anchored to lie on a circle with radius 20.

.. image:: files/anchor_constraint.gif
	:width: 50 %

See 
`this notebook <https://github.com/cvxgrp/pymde/blob/main/examples/anchor_constraints.ipynb>`_
for the code to make this embedding (and GIF).

Custom constraints
^^^^^^^^^^^^^^^^^^

It is possible to specify a custom constraint set. To learn how to 
do so, consult the :ref:`API documentation <api_constraints>`.

Computing embeddings
--------------------

After creating an MDE problem, you can compute an embedding by calling
the its :any:`embed <pymde.MDE.embed>` method. The embed method takes
a few optional hyper-parameters. Here is its documentation.

.. automethod:: pymde.MDE.embed
   :noindex:

Computing an embedding saves some statistics in the
:any:`solve_stats <pymde.optim.SolveStats>` attribute.

Sanity-checking embeddings
--------------------------
The MDE framework gives you a few ways to sanity-check embeddings.

Plotting embeddings
^^^^^^^^^^^^^^^^^^^
If your embedding is in three or fewer dimensions, the first thing to do
(after calling the ``embed`` method) is to simply plot it with :any:`pymde.plot`,
and color it by some attributes that were not used in the embedding process.
You can optionally pass in a list of edges to this function, which will
superimpose edges onto the scatter plot. Read the API documentation for
more details.

GIFs can be created with the :any:`pymde.MDE.play` method.

The CDF of distortions
^^^^^^^^^^^^^^^^^^^^^^

Regardless of the embedding dimension, the next thing to do is to
plot the cumulative distribution function (CDF) of distortions. You can
do this by calling the ``distortions_cdf`` method on an MDE instance:

.. code:: python3

   mde.distortions_cdf()

This will result in a plot like

.. image:: /files/distortions_cdf.png

In this particular case, we see that most distortions are very small, but
roughly 10 percent of them are much larger. This means that embedding
the items was "easy", except for these 10 percent of edges.

Outliers
^^^^^^^^

Next, you should manually inspect the items in, say, the 10 most highly
distorted pairs; this is similar to debugging a supervised learning model by
examining its mistakes. You can get the list of edges sorted from most
distorted to least like so:

.. code:: python3

   pairs, distortions = mde.high_distortion_pairs().
   highly_distorted_pairs = pairs[:10]

In the case of a specific embedding of MNIST, some of these pairs ended up
containing oddly written digits, while others looked like they shouldn't
have been paired:

.. image:: /files/mnist_outliers.png

After inspecting the highly distorted pairs, you have a few options.
You can leave your embedding as is, if you think your embedding is reasonable;
you can throw out some of the highly distorted edges if you think they don't
belong; you can modify your distortion functions to be less sensitive 
to large distances; or you can even remove some items from your original
dataset, if they appear malformed.

Comparing embeddings
--------------------
Suppose you want to compare two different embeddings, which have the same
number of items and the same embedding dimension. If you have an MDE instance,
you can evaluate the average distortion of each embedding by calling
its ``average_distortion`` method.

It can also be meaningful to compute a distance between two embeddings. The
average distortion is invariant to rotations and reflections of embeddings,
so two embeddings must first be *aligned* before they can be compared.

To align one embedding to another, use the :any:`pymde.align` function:

.. code:: python3

   aligned_embedding = pymde.align(source=embedding, target=another_embedding)

This function rotates and reflects the source embedding to be as close to 
the target embedding as possible, and returns this rotated embedding. After
aligning, you can compare embeddings by plotting them (if the dimension is 3 or
less), or by computing the Frobenius norm of their difference (this distance
will make sense if both embeddings are standardized, since that will put
them on the same scale, but it will make less sense otherwise).

Embedding new points
--------------------

Suppose we have embedded some number of items, and later we obtain additional
items of the same type that we wish to embed. For example, we might
have embedded the MNIST dataset, and later we obtain more images we'd like
to embed.

Often we want to embed
the new items without changing the vectors for the old data. To do so,
we can solve a small MDE problem involving the new items and some of the old
ones: some edges will be between new items, and importantly some edges will
connect the new items to old items. The old items can be held in place
with an anchor constraint.

For example, here is how to update an embedding of MNIST.

.. code:: python3

  import pymde
  import torch

  mnist = pymde.datasets.MNIST()

  n_train = 35000
  train_data = mnist.data[:n_train]
  val_data = mnist.data[n_train:]

  train_embedding = pymde.preserve_neighbors(
      train_data, verbose=True).embed(verbose=True)
  updated_embedding = pymde.preserve_neighbors(
      torch.vstack([train_data, val_data]),
      constraint=pymde.Anchored(torch.arange(n_train), train_embedding),
      verbose=True).embed(verbose=True)

A complete example is provided in the below notebook.

- `Updating embedding notebook <https://github.com/cvxgrp/pymde/blob/main/examples/updating_an_existing_embedding.ipynb>`_
