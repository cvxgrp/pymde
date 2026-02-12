# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "pymde",
#     "torch",
# ]
# ///

import marimo

__generated_with = "0.19.10"
app = marimo.App()

with app.setup:
    import math
    import marimo as mo
    import pymde
    import torch


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Anchor Constraints

    In this notebook, we show how to use an [anchor constraint](https://pymde.org/mde/index.html#anchored), which
    lets us pin some of the embedding vectors to specific values. This can be useful if you know a priori what some of the embedding vectors should be.

    Below, we'll use an anchor constraint when laying out a binary tree: we'll pin the leaves to lie on a circle.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This next cell constraints a binary tree with a specified depth. You can change the depth of the tree by changing the value of the `depth` variable.
    """)
    return


@app.cell
def _():
    depth = 9
    n_items = 2**(depth + 1) - 1

    edges = []
    stack = [0]
    while stack:
        root = stack.pop()
        first_child = root*2 + 1
        second_child = root*2 + 2
        if first_child < n_items:
            edges.append((root, first_child))
            stack.append(first_child)
        if second_child < n_items:
            edges.append((root, second_child))
            stack.append(second_child)

    tree = pymde.Graph.from_edges(torch.tensor(edges))
    return depth, edges, n_items, tree


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To layout the graph, we're going to preserve the shortest path distances between all pairs of nodes. The next cell computes these distances.
    """)
    return


@app.cell
def _(tree):
    shortest_paths_graph = pymde.preprocess.graph.shortest_paths(tree)
    return (shortest_paths_graph,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To use an anchor constraint, we need the indices of the items we want to pin (remember that we will pin the leaves). We get the indices of the leaves using a simple formula.
    """)
    return


@app.cell
def _(depth):
    # these are the indices of the nodes that we will pin in place
    leaves = torch.arange(2**depth) + 2**depth - 1
    return (leaves,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We want to pin the leaves to be uniformly spaced on a circle. The below cell constructs a Tensor in which the `i`th row holds the location to which the `i`th leaf should be pinned.

    If you like, you can try changing the `radius` to see what effect it has on the embedding.
    """)
    return


@app.cell
def _(leaves):
    radius = 20

    # pin the root to be at (0, 0), and the leaves to be spaced uniformly on a circle
    angles = torch.linspace(0, 2*math.pi, leaves.numel() + 1)[1:]
    positions = radius * torch.stack([torch.sin(angles), torch.cos(angles)], dim=1)
    positions
    return (positions,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now that we have the leaves (the anchors) and the desired positions (their values), we can create the anchor constraint.
    """)
    return


@app.cell
def _(leaves, positions):
    anchor_constraint = pymde.Anchored(anchors=leaves, values=positions)
    return (anchor_constraint,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can now create the embedding. Feel free to play around with the distortion function (try changing it to `pymde.losses.Quadratic`, or `pymde.losses.Cubic`) to see what effect the choice of distortion function has on the emedding.
    """)
    return


@app.cell
def _(anchor_constraint, edges, n_items, shortest_paths_graph):
    mde = pymde.MDE(
        n_items,
        embedding_dim=2,
        edges=shortest_paths_graph.edges,
        distortion_function=pymde.losses.WeightedQuadratic(shortest_paths_graph.distances),
        constraint=anchor_constraint,
    )
    mde.embed(snapshot_every=1, verbose=True)
    mde.plot(edges=edges)
    return (mde,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For fun, we can make a GIF of how the embedding was created, using the `play` method.
    """)
    return


@app.cell
def _(edges, mde):
    mde.play(edges=edges)
    return


if __name__ == "__main__":
    app.run()
