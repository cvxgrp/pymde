# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ipython==9.10.0",
#     "marimo",
#     "pymde",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np
    import pymde
    import scipy.sparse as sp
    import torch


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Drawing Graphs


    This notebook shows how to use `PyMDE` to layout graphs, using simple MDE problems.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Creating a graph

    The `PyMDE.Graph` class encapsulates an undirected weighted graph.

    Graphs have three parts: nodes, edges, and weights.

    - *Nodes* are represented by non-negative integers; a graph on $n$ nodes will have nodes $0, 1, 2, \ldots, n - 1$.

    - An *edge* is an array `e` of two node indices, with `e[0] < e[1]`.

    - Each edge has an associated scalar number, which can be interpreted as a *weight* on the edge or a distance between the nodes it connects.

    Graphs can be constructed from adjacency matrices or adjacency lists.
    - An *adjacency matrix* is a sparse matrix `A` of shape `(n, n)`, with `A[i, j]` giving the weight associated with edge $(i, j)$. If `A[i, j]` is `0`, then edge $(i, j)$ is not present in the grpah.
    - An *adjacency list* is an array of edges, with each row corresponding to an edge.

    Here is a simple example that show how to create graphs.
    """)
    return


@app.cell
def _():
    adjacency_matrix = sp.csr_matrix(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
    _triangle = pymde.Graph(adjacency_matrix)
    _triangle.draw()
    return


@app.cell
def _():
    _edges = torch.tensor([[0, 1], [0, 2], [1, 2]])
    _triangle = pymde.Graph.from_edges(_edges)
    _triangle.draw()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Drawing graphs

    The `draw` method lays out the graph in the Cartesian plane. Behind-the-scenes, it constructs and solves
    MDE problems that are suitable for laying out graphs.


    We used this method in the previous cells. In the remainder of this notebook, we show how to draw graphs by
    creating and solving MDE problems from scratch.
    """)
    return


@app.cell
def _():
    n_items = 20
    embedding_dim = 2
    return embedding_dim, n_items


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Complete Graph
    """)
    return


@app.cell
def _(embedding_dim, n_items):
    _edges = pymde.all_edges(n_items)
    mde = pymde.MDE(
        n_items,
        embedding_dim=embedding_dim,
        edges=_edges,
        distortion_function=pymde.penalties.Cubic(weights=1.0),
        constraint=pymde.Standardized(),
    )
    mde.embed()
    mde.plot(edges=_edges)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Chain Graph
    """)
    return


@app.cell
def _(embedding_dim, n_items):
    chain_graph = pymde.Graph.from_edges(
        torch.tensor([(i, i + 1) for i in range(n_items - 1)])
    )
    _shortest_paths_graph = pymde.preprocess.graph.shortest_paths(chain_graph)
    mde_1 = pymde.MDE(
        n_items,
        embedding_dim,
        _shortest_paths_graph.edges,
        pymde.losses.WeightedQuadratic(deviations=_shortest_paths_graph.distances),
    )
    mde_1.embed()
    mde_1.plot(edges=chain_graph.edges)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Cycle Graph
    """)
    return


@app.cell
def _(embedding_dim, n_items):
    cycle_graph = pymde.Graph.from_edges(
        torch.tensor([(i, i + 1) for i in range(n_items - 1)] + [(n_items - 1, 0)])
    )
    _shortest_paths_graph = pymde.preprocess.graph.shortest_paths(cycle_graph)
    mde_2 = pymde.MDE(
        n_items,
        embedding_dim,
        _shortest_paths_graph.edges,
        pymde.losses.WeightedQuadratic(_shortest_paths_graph.distances),
    )
    mde_2.embed()
    mde_2.plot(edges=cycle_graph.edges)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Star Graph
    """)
    return


@app.cell
def _(embedding_dim, n_items):
    star_graph = pymde.Graph.from_edges(
        torch.tensor([(0, i) for i in range(1, n_items)])
    )
    _shortest_paths_graph = pymde.preprocess.graph.shortest_paths(star_graph)
    mde_3 = pymde.MDE(
        n_items,
        embedding_dim,
        _shortest_paths_graph.edges,
        pymde.losses.WeightedQuadratic(_shortest_paths_graph.distances),
    )
    mde_3.embed()
    mde_3.plot(edges=star_graph.edges)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Tree
    """)
    return


@app.cell
def _(embedding_dim, n_items):
    _edges = []
    stack = [0]
    while stack:
        root = stack.pop()
        c1 = root * 2 + 1
        c2 = root * 2 + 2
        if c1 < n_items:
            _edges.append((root, c1))
            stack.append(c1)
        if c2 < n_items:
            _edges.append((root, c2))
            stack.append(c2)
    tree = pymde.Graph.from_edges(torch.tensor(_edges))
    _shortest_paths_graph = pymde.preprocess.graph.shortest_paths(tree)
    mde_4 = pymde.MDE(
        n_items,
        embedding_dim,
        _shortest_paths_graph.edges,
        pymde.losses.WeightedQuadratic(_shortest_paths_graph.distances),
    )
    mde_4.embed(snapshot_every=1, max_iter=20)
    mde_4.plot(edges=tree.edges)
    return mde_4, tree


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The `play` method creates a movie showing how the embedding was made. Try it out!
    """)
    return


@app.cell
def _(mde_4, tree):
    mde_4.play(edges=tree.edges, fps=15.0, savepath="/tmp/graph.gif")
    mo.image("/tmp/graph.gif")
    return


if __name__ == "__main__":
    app.run()
