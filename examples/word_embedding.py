# /// script
# dependencies = ["plotly"]
# ///

import marimo

__generated_with = "0.19.10"
app = marimo.App()

with app.setup:
    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import pymde
    import torch


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Word embedding

    This notebook shows how to use PyMDE to create word embeddings in a few lines of code.

    We will embed the 5000 most popular academic interests listed in Google Scholar.
    """)
    return


@app.cell
def _():
    np.random.seed(0)
    torch.manual_seed(0)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Cooccurrence matrix

    The data is a **cooccurrence matrix**, which counts the number of times each pair of interests cooccurred in a researchers Google Scholar profile page.

    The ith row (and column) of the matrix corresponds to the ith interest (i.e., `interests[i]`)
    """)
    return


@app.cell
def _():
    dataset = pymde.datasets.google_scholar_interests()

    cooccurrence_graph = dataset.data
    interests = dataset.attributes['interests']
    return cooccurrence_graph, interests


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can visualize the cooccurrence matrix. No patterns are immediately discernible.
    """)
    return


@app.cell
def _(cooccurrence_graph):
    plt.figure(figsize=(12, 12))
    plt.spy(cooccurrence_graph.A, ms=0.01, color='k')
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    As a first cut, we will embed the words into $\mathbf{R}^1$, using PCA.
    """)
    return


@app.cell
def _(cooccurrence_graph):
    pca_embedding = pymde.pca(cooccurrence_graph.A.todense(), 1)
    pca_embedding
    return (pca_embedding,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can visualize the PCA embedding by sorting the cooccurrence matrix according to the indices that sort the PCA embedding. Notice that this reveals some interesting structure.
    """)
    return


@app.cell
def _(cooccurrence_graph, pca_embedding):
    sort_indices = pca_embedding.flatten().sort().indices.numpy()
    A_sorted = cooccurrence_graph.A[sort_indices][:, sort_indices]

    plt.figure(figsize=(12, 12))
    plt.spy(A_sorted, ms=1e-2, color='k')
    plt.gca()
    return (sort_indices,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here are the first 20 entries in the sorted cooccurrence matrix. We can see that theey are mostly about artificial intelligence and biology, meaning that our simple embedding has uncovered that these topics are related.
    """)
    return


@app.cell
def _(interests, sort_indices):
    list(np.array(interests)[sort_indices])[:20]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Embedding

    Now we will construct a more sophisticated embedding. We start by converting the counts into dissimilarity scores,
    then compute a shortest path metric with respect to these scores.
    """)
    return


@app.cell
def _(cooccurrence_graph):
    dissimilarities = 1. / torch.log(cooccurrence_graph.weights)
    dissimilarity_graph = pymde.Graph.from_edges(cooccurrence_graph.edges, dissimilarities)
    return (dissimilarity_graph,)


@app.cell
def _(dissimilarity_graph):
    shortest_path_graph = pymde.preprocess.graph.shortest_paths(dissimilarity_graph, verbose=True, n_workers=6)
    return (shortest_path_graph,)


@app.cell
def _(interests):
    def interactive(X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        fig = go.Figure(data=go.Scatter(x=X[:, 0],
                                        y=X[:, 1],
                                        marker_size=2.,
                                        mode='markers',
                                        text=interests))
        fig.show()

    return (interactive,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Finally we compute an embedding that preserves neighbors under the shortest path metric.
    """)
    return


@app.cell
def _(shortest_path_graph):
    mde = pymde.preserve_neighbors(shortest_path_graph, verbose=True)
    return (mde,)


@app.cell
def _(mde):
    X = mde.embed(verbose=True)
    return (X,)


@app.cell
def _(mde):
    mde.plot(colors=['black'])
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Try mousing over this embedding, to reveal the interest labels for each node. You'll find that the embedding has done a good job of clustering similar interests together.
    """)
    return


@app.cell
def _(X, interactive):
    interactive(X)
    return


if __name__ == "__main__":
    app.run()
