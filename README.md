# PyMDE
![](https://github.com/cvxgrp/pymde/workflows/Test/badge.svg) ![](https://github.com/cvxgrp/pymde/workflows/Deploy/badge.svg)

PyMDE is a Python library for computing vector embeddings for finite sets of
items, such as images, biological cells, nodes in a network, or any other
abstract object.

What sets PyMDE apart from other embedding libraries is that it provides a
simple but general framework for embedding, called _Minimum-Distortion
Embedding_ (MDE). With MDE, it is easy to recreate well-known embeddings and to
create new ones, tailored to your particular application. PyMDE is competitive
in runtime with more specialized embedding methods. With a GPU, it can be
even faster.


PyMDE can be enjoyed by beginners and experts alike. It can be used to:
* reduce the dimensionality of vector data;
* generate feature vectors;
* visualize datasets, small or large;
* draw graphs in an aesthetically pleasing way;
* find outliers in your original data;
* and more.


PyMDE is very young software, under active development. If you run into issues,
or have any feedback, please reach out by [filing a Github
issue](https://github.com/cvxgrp/pymde/issues).

This README is a crash course on how to get started with PyMDE. The full
documentation is still under construction.

- [Installation](#installation)
- [Getting started](#getting-started)
- [MDE](#mde)
- [Preprocessing data](#preprocessing-data)
- [Example notebooks](#example-notebooks)
- [Citing](#citing)


## Installation
PyMDE is available on the Python Package Index. Install it with 

```
pip install pymde
```

## Getting started
Getting started with PyMDE is easy. For embeddings that work out-of-the box, we provide two main functions:

```python3
pymde.preserve_neighbors
```

which preserves the local structure of original data, and 

```python3
pymde.preserve_distances
```

which preserves pairwise distances or dissimilarity scores in the original
data.

**Arguments.** The input to these functions is the original data, represented
either as a data matrix in which each row is a feature vector, or as a
(possibly sparse) graph encoding pairwise distances. The embedding dimension is
specified by the `embedding_dim` keyword argument, which is `2` by default.

**Return value.** The return value is an `MDE` object. Calling the `embed()`
method on this object returns an **embedding**, which is a matrix
(`torch.Tensor`) in which each row is an embedding vector. For example, if the
original input is a data matrix of shape `(n_items, n_features)`, then the
embedding matrix has shape `(n_items, embeddimg_dim)`.

We give examples of using these functions below. 

### Preserving neighbors
The following code produces an embedding of the MNIST dataset (images of
handwritten digits), in a fashion similar to LargeVis, t-SNE, UMAP, and other
neighborhood-based embeddings. The original data is a matrix of shape `(70000,
784)`, with each row representing an image.

```python3
import pymde

mnist = pymde.datasets.MNIST()
embedding = pymde.preserve_neighbors(mnist.data).embed()
pymde.plot(embedding, color_by=mnist.attributes['digits'])
```

![](https://github.com/cvxgrp/pymde/blob/main/images/mnist.png?raw=true)

Unlike most other embedding methods, PyMDE can compute embeddings that satisfy
constraints. For example:

```python3
embedding = pymde.preserve_neighbors(mnist.data, constraint=pymde.Standardized()).embed()
pymde.plot(embedding, color_by=mnist.attributes['digits'])
```

![](https://github.com/cvxgrp/pymde/blob/main/images/mnist_std.png?raw=true)

The standardization constraint enforces the embedding vectors to be centered
and have uncorrelated features.


### Preserving distances
The function `pymde.preserve_distances` is useful when you're more interested
in preserving the gross global structure instead of local structure. 

Here's an example that produces an embedding of an academic coauthorship
network, from Google Scholar. The original data is a sparse graph on roughly
40,000 authors, with an edge between authors who have collaborated on at least
one paper.

```python3
import pymde

google_scholar = pymde.datasets.google_scholar()
embedding = pymde.preserve_distances(google_scholar.data).embed()
pymde.plot(embedding, color_by=google_scholar.attributes['coauthors'], color_map='viridis', background_color='black')
```

![](https://github.com/cvxgrp/pymde/blob/main/images/scholar.jpg?raw=true)

More collaborative authors are colored brighter, and are near the center of the
embedding.


## MDE
The functions `pymde.preserve_neighbors` and `pymde.preserve_distances` from
the previous section both returned instances of the `MDE` class.  For simple
use cases, you can just call the `embed()` method on the returned instance and
use the resulting embedding. 

For more advanced use cases, such as creating custom embeddings, debugging
embeddings, and looking for outliers, you'll need to understand what an `MDE`
object represents.

We'll go over the basics of the `MDE` class in this section.

### Hello, World
An MDE instance represents an "MDE problem", whose solution is an embedding. An MDE problem has 5 parts:
* the number of items being embedded
* the embedding dimension
* a list of edges
* distortion functions
* a constraint


We'll explain what these parts are using a simple example: embedding the nodes
of a cycle graph into 2 dimensions.

**Number of items.** We'll embed a cycle graph on 5 nodes. We have 5 items (the
nodes we are embedding), labeled `0`, `1`, `2`, `3`, and `4`.

**Embedding dimension.** The embedding dimension is 2.

**Edges.** Every MDE problem has a list of  edges (i.e., pairs) `(i, j)`, `0 <=
i < j < n_items`. If an edge `(i, j)` is in this list, it means we have an
opinion on the relationship between items `i` and `j` (e.g., we know that they
are similar). 

In our example, we are embedding a cycle graph, and we know that adjacent nodes
are similar. We choose the edges as

```python3
edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]])
```

Notice that `edges` is a `torch.Tensor` of shape `(n_edges, 2)`.

**Distortion functions.** The relationships between items are articulated by
distortion functions (or loss functions). Each distortion function is
associated with one pair `(i, j)`, and it maps the Euclidean distance `d_{ij}`
between the embedding vectors for items `i` and `j` to a scalar _distortion_
(or loss). The goal of an MDE problem is to find an embedding that minimizes
the average distortion, across all edges.

In this example, for each edge `(i, j)`, we'll use the distortion function
`w_{ij} d_{ij}**3`, where `w_{ij}` is a scalar weight. We'll set all the
weights to one for simplicity. This means we'll penalize large distances, and
treat all edges equally.

```python3
distortion_function = pymde.penalties.Cubic(weights=torch.ones(edges.shape[0]))
```

In general, the constructor for a penalty takes a `torch.Tensor` of weights,
with `weights[k]` giving the weight associated with `edges[k]`.

**Constraint.** In our simple example, the embedding will collapse to 0 if it
is unconstrained (since all the weights are positive). We need a constraint
that forces the embedding vectors to spread out.

The constraint we will impose is called a _standardization constraint_. This
constraint requires the embedding vectors to be centered, and to have
uncorrelated feature columns. This forces the embedding to spread out.

We create the constraint with:

```python3
constraint = pymde.Standardized()
```

**Creating the MDE problem.** Having specified the number of items, embedding
dimension, edges, distortion function, and constraint, we can now create the
embedding problem:

```python3
import pymde

mde = pymde.MDE(
  n_items=5,
  embedding_dim=2,
  edges=torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]]),
  distortion_function=pymde.penalties.Cubic(weights=torch.ones(edges.shape[0])),
  constraint=pymde.Standardized())
```

**Computing an embedding.** Finally, we can compute the embedding by calling
the `embed` method:

```python3
embedding = mde.embed()
print(embedding)
```
prints

```
tensor([[ 0.5908, -1.2849],
        [-1.0395, -0.9589],
        [-1.2332,  0.6923],
        [ 0.2773,  1.3868],
        [ 1.4046,  0.1648]])
```

(This method returns the embedding, and it also saves it in the `X` attribute
of the `MDE` instance (`mde.X`).)

Since this embedding is two-dimensional, we can also plot it:

```python3
mde.plot(edges=mde.edges)
```

![](https://github.com/cvxgrp/pymde/blob/main/images/cycle.jpg?raw=true)

The embedding vectors are the dots in this image. Notice that we passed the
list of edges to the `plot` method, to visualize each edge along with the
vectors.

**Summary.** This was a toy example, but it showed all the main parts of an MDE problem:
* `n_items`: the number of things ("items") being embedded
* `embedding_dim`: the dimension of the embedding (e.g., 1, 2, or 3 for visualization, or larger for feature generation)
* `edges`: a `torch.Tensor`, of shape `(n_edges, 2)`, in which each row is a pair `(i, j)`.
* `distortion_function`: a vectorized distortion function, mapping a `torch.Tensor` containing the `n_edges` embedding distances to a `torch.Tensor` of distortions of the same length.
* `constraint`: a constraint instance, such as `pymde.Standardized()`, or `None` for unconstrained embeddings.

Next, we'll learn more about distortion functions, including how to write
custom distortion functions.

### Distortion functions
The vectorized distortion function is just any callable that maps a
`torch.Tensor` of embedding distances to a `torch.Tensor` of distortions, using
PyTorch operations. The callable must have the signature

```python3
distances: torch.Tensor[n_edges] -> distortions: torch.Tensor[n_edges]
```

Both the input and the return value must have shape `(n_edges,)`, where
`n_edges` is the number of edges in the MDE problem. The entry `distances[k]`
is the embedding distance corresponding to the `k`th edge, and the entry
`distortions[k]` is the distortion for the `kth` edge.

As a simple example, the cubic penalty we used in the Hello, World example can
be implemented as

```python3
def distortion_function(distances):
    return distances ** 3
```

Distortion functions usually come as one of two types: they are either derived
from scalar **weights** (one weight per edge) or **deviations** (also one per
edge). PyMDE provides a library of distortion functions derived from weights,
as well as a library of distortion functions derived from original deviations.
These functions are called **penalties** and **losses** respectively.

**Penalties.** Distortion functions derived from weights have the general form
`distortion[k] := w[k] penalty(distances[k])`, where `w[k]` is the weight
associated with the k-th edge and `penalty` is a penalty function. Penalty
functions are increasing functions. A positive weight
indicates that the items in the `k`th edge are similar, and a negative weight
indicates that they are dissimilar. The larger the magnitude of the weight, the
more similar or dissimilar the items are.

A number of penalty functions are available in the `pymde.penalties` module.

**Losses.** Distortion functions derived from original deviations have the
general form `distortion[k] = loss(deviations[k], distances[k])`, where
`deviations[k]` is a positive number conveying how dissimilar the items in
`edges[k]` are; the larger `deviations[k]` is, the more dissimilar the items
are.  The loss function is 0 when `deviations[k] == distances[k]`, and
nonnegative otherwise.

A number of loss functions are available in the `pymde.losses` module.

## Preprocessing data
The `preserve_neighbors` and `preserve_distances` functions create MDE problems
based on weights and original deviations respectively, preprocessing the
original data to obtain the weights or distances.

A number of preprocessors are provided in the `pymde.preprocess` module.

## Example notebooks
The best way to learn more about PyMDE is to try it out, using the
[example notebooks](https://github.com/cvxgrp/pymde/examples) as a starting point.

## Citing
To cite our work, please use the following BibTex entry.

```
@article{agrawal2020minimum,
  author  = {Agrawal, Akshay and Ali, Alnur and Boyd, Stephen},
  title   = {Minimum-Distortion Embedding},
  year    = {2020},
}
```
