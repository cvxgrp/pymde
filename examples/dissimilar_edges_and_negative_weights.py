# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "pymde",
#     "matplotlib",
#     "numpy",
#     "torch",
# ]
# ///

import marimo

__generated_with = "0.19.10"
app = marimo.App()

with app.setup:
    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import pymde
    import torch


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Dissimilar edges and negative weights

    In this notebook, we show how the number of repulsive edges and the magnitude of the negative weights
    can affect the embeddings, when creating embeddings based on weights. We use MNIST as a case study.

    The summary of what we'll find: the more dissimilar edges (and the larger the magnitude of the negative weight), the more uniformly spread out the embedding is. The fewer edges (and the smaller the magnitude of the weight), the more tightly clustered the embedding is. These comments are for when a standardization constraint is enforced. Without the constraint, the embedding can collapse to 0 if there are too few dissimilar edges.

    You should make sure to look at the MNIST notebook before going through this one.
    """)
    return


@app.cell
def _():
    mnist = pymde.datasets.MNIST()

    images = mnist.data
    digits = mnist.attributes['digits']
    return digits, images


@app.cell
def _(images):
    knn_graph = pymde.preprocess.k_nearest_neighbors(images, k=15, verbose=True)
    return (knn_graph,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Quadratic MDE problem
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We start by computing a spectral initialization.
    """)
    return


@app.cell
def _(images, knn_graph):
    quadratic_mde = pymde.MDE(
        n_items=images.shape[0],
        embedding_dim=2,
        edges=knn_graph.edges,
        distortion_function=pymde.penalties.Quadratic(knn_graph.weights),
        constraint=pymde.Standardized())
    quadratic_mde.embed(verbose=True)
    return (quadratic_mde,)


@app.cell
def _(digits, quadratic_mde):
    quadratic_mde.plot(color_by=digits)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Dissimilar edges
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    First we vary the number of dissimilar edges, and show what effect this has on the embedding.
    """)
    return


@app.cell
def _(images, knn_graph):
    def get_edges(n_dissimilar_edges, neg_weight=-1.0):
        similar_edges = knn_graph.edges
        dissimilar_edges = pymde.preprocess.dissimilar_edges(images.shape[0], num_edges=n_dissimilar_edges, similar_edges=similar_edges)
        _edges = torch.cat([similar_edges, dissimilar_edges])
        _weights = torch.cat([knn_graph.weights, neg_weight * torch.ones(dissimilar_edges.shape[0])])
        return (_edges, _weights)

    return (get_edges,)


@app.cell
def _(digits, get_edges, images, knn_graph, quadratic_mde):
    n_similar = knn_graph.edges.shape[0]
    n_dissimilar = [n_similar // 8, n_similar // 4, n_similar // 2, n_similar, 2 * n_similar, 4 * n_similar]
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for n_dissimilar_edges in n_dissimilar:
        _edges, _weights = get_edges(n_dissimilar_edges)
        print('# dissimilar edges / total: ', n_dissimilar_edges / _edges.shape[0])
        _f = pymde.penalties.PushAndPull(weights=_weights, attractive_penalty=pymde.penalties.Log1p, repulsive_penalty=pymde.penalties.Log).to(_device)
        _std_mde = pymde.MDE(n_items=images.shape[0], embedding_dim=2, edges=_edges.to(_device), distortion_function=_f, constraint=pymde.Standardized())
        _std_mde.embed(X=quadratic_mde.X.to(_device))
        _std_mde.plot(color_by=digits)
        plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Negative weights

    Next we vary the weights instead of the number of edges. The effect is similar to what happened when we varied the dissimilar edges.
    """)
    return


@app.cell
def _(digits, get_edges, images, knn_graph, quadratic_mde):
    neg_weights = [-1 / 8, -1 / 4, -1 / 2, -1.0, -2.0, -4.0]
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for neg_weight in neg_weights:
        print('negative weight: ', neg_weight)
        _edges, _weights = get_edges(knn_graph.edges.shape[0], neg_weight)
        _f = pymde.penalties.PushAndPull(weights=_weights, attractive_penalty=pymde.penalties.Log1p, repulsive_penalty=pymde.penalties.Log).to(_device)
        _std_mde = pymde.MDE(images.shape[0], embedding_dim=2, edges=_edges.to(_device), distortion_function=_f, constraint=pymde.Standardized())
        _std_mde.embed(X=quadratic_mde.X.to(_device))
        _std_mde.plot(color_by=digits)
        plt.show()
    return


if __name__ == "__main__":
    app.run()
