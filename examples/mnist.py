# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "pandas==3.0.1",
#     "pymde==0.3.0",
#     "torch==2.10.0",
# ]
# ///

import marimo

__generated_with = "0.20.1"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)

with app.setup:
    import matplotlib.pyplot as plt
    import pandas as pd
    import pymde
    import torch
    import marimo as mo

    mnist = pymde.datasets.MNIST()


@app.function
def compute_embedding(embedding_dim, constraint, algorithm):
    mo.output.append(
        mo.md("Your embedding is being computed ... hang tight!").callout(kind="warn")
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if algorithm == "log1p":
        X = pymde.preserve_neighbors(
            mnist.data,
            embedding_dim=embedding_dim,
            constraint=constraint,
            device=device,
            verbose=True,
        ).embed(verbose=True)
    elif algorithm == "umap":
        X = pymde.preserve_neighbors(
            mnist.data,
            embedding_dim=embedding_dim,
            constraint=pymde.Centered(),
            device=device,
            verbose=True,
        ).embed(verbose=True)
    elif algorithm == "pca":
        X = pymde.pca(mnist.data, 2)
    else:
        X = pymde.laplacian_embedding(mnist.data, embedding_dim=2).embed(verbose=True)
    mo.output.clear()
    return X


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # MNIST

    This notebook shows how to use the function `pymde.preserve_neighbors` to produce embeddings that highlight the local structure of your data, using MNIST as a case study. In these embeddings similar digits are near each other, and dissimilar digits are not near each other.

    It also shows how to debug embeddings and search for outliers in your original data, using PyMDE.
    """)
    return


@app.cell
def _():
    algorithm = mo.ui.dropdown(["log1p", "pca", "laplacian", "umap"], value="log1p")
    algorithm
    return (algorithm,)


@app.cell
def _(algorithm, constraint, embedding_dimension):
    with mo.persistent_cache("embedding"):
        embedding = compute_embedding(embedding_dimension, constraint, algorithm.value)

    embedding = pymde.rotate(embedding, 40) * torch.tensor([-1, -1])
    return (embedding,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    Here's an **embedding of MNIST**: each point represents a digit,
    with similar digits close to each other.

    **Try making a selection with your mouse!**
    """)
    return


@app.cell
def _(embedding):
    ax = pymde.plot(embedding, color_by=mnist.attributes["digits"])
    fig = mo.ui.matplotlib(ax)
    fig
    return (fig,)


@app.cell
def _(embedding, fig):
    mask = fig.value.get_mask(embedding[:, 0], embedding[:, 1])
    return (mask,)


@app.cell(hide_code=True)
def _(mask, table):
    # mo.stop() prevents this cell from running if the chart has
    # no selection
    mo.stop(not mask.any())

    # show 10 images: either the first 10 from the selection, or the first ten
    # selected in the table
    selected_images = (
        show_images(list(mask.nonzero()[0]))
        if not len(table.value)
        else show_images(list(table.value["index"]))
    )

    mo.md(
        f"""
        **Here's a preview of the images you've selected**:

        {mo.as_html(selected_images)}

        Here's all the data you've selected.

        {table}
        """
    )
    return


@app.cell
def _(df, mask):
    table = mo.ui.table(df[mask])
    return (table,)


@app.cell
def _():
    embedding_dimension = 2
    constraint = pymde.Standardized()
    return constraint, embedding_dimension


@app.cell
def _(embedding):
    indices = torch.arange(mnist.data.shape[0]).numpy()

    df = pd.DataFrame(
        {
            "index": indices,
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "digit": mnist.attributes["digits"][indices],
        }
    )
    return (df,)


@app.function
def show_images(indices, max_images=10):
    indices = indices[:max_images]
    images = mnist.data.reshape((-1, 28, 28))[indices]
    fig, axes = plt.subplots(1, len(indices))
    fig.set_size_inches(12.5, 1.5)
    if len(indices) > 1:
        for im, ax in zip(images, axes.flat):
            ax.imshow(im, cmap="gray")
            ax.set_yticks([])
            ax.set_xticks([])
    else:
        axes.imshow(images[0], cmap="gray")
        axes.set_yticks([])
        axes.set_xticks([])
    plt.tight_layout()
    return fig


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Embeddings, in one line
    """)
    return


@app.cell
def _():
    _embedding = pymde.preserve_neighbors(mnist.data).embed(verbose=True)
    pymde.plot(_embedding, color_by=mnist.attributes["digits"])
    return


@app.cell
def _():
    _embedding = pymde.preserve_neighbors(
        mnist.data, constraint=pymde.Standardized()
    ).embed(verbose=True)
    pymde.plot(_embedding, color_by=mnist.attributes["digits"])
    return


@app.cell
def _():
    _embedding = pymde.preserve_neighbors(
        mnist.data,
        attractive_penalty=pymde.penalties.Quadratic,
        repulsive_penalty=None,
    ).embed(verbose=True)
    pymde.plot(_embedding, color_by=mnist.attributes["digits"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Embeddings, from scratch
    """)
    return


@app.cell
def _():
    knn_graph = pymde.preprocess.k_nearest_neighbors(
        mnist.data, k=15, verbose=True
    )
    return (knn_graph,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Quadratic MDE problems
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The `embed` method computes and returns an embedding.

    The embedding is also accessible via the `X` attribute of the `MDE` instance.
    """)
    return


@app.cell
def _(knn_graph):
    quadratic_mde = pymde.MDE(
        n_items=mnist.data.shape[0],
        embedding_dim=2,
        edges=knn_graph.edges,
        distortion_function=pymde.penalties.Quadratic(knn_graph.weights),
        constraint=pymde.Standardized(),
    )

    quadratic_mde.embed(verbose=True)
    return (quadratic_mde,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    After embedding, we can call the `plot` method to visualize the embedding (when `embedding_dim` $\leq 3$).

    The `color_by` keyword argument takes a length-$n$ list of attribute values associated with the items; the values are used to color the points, with each unique value getting its own color.

    Here, the attribute is the digit depicted by the image.
    """)
    return


@app.cell
def _(quadratic_mde):
    quadratic_mde.plot(color_by=mnist.attributes["digits"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Next, we fit an embedding with 3 dimensions. We can visualize 3D embeddings with `PyMDE`.
    """)
    return


@app.cell
def _(knn_graph):
    quadratic_mde_3d = pymde.MDE(
        n_items=mnist.data.shape[0],
        embedding_dim=3,
        edges=knn_graph.edges,
        distortion_function=pymde.penalties.Quadratic(knn_graph.weights),
        constraint=pymde.Standardized(),
    )

    quadratic_mde_3d.embed(verbose=True)
    quadratic_mde_3d.plot(color_by=mnist.attributes["digits"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Debugging the embedding
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The distribution of distortions

    We can visualize the cumulative distribution of distortions. We see that a small number of pairs are responsible for most of the embedding's distortion.
    """)
    return


@app.cell
def _(quadratic_mde):
    quadratic_mde.distortions_cdf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### High distortion pairs

    It can be instructive to inspect the pairs with highest distortion. A pair, here one column of the grid, is a set of two images that our MDE problem was told are similar. In this case, some of the high-distortion pairs are oddly drawn digits, and don't appear very similar to each other.
    """)
    return


@app.cell
def _(quadratic_mde):
    pairs, distortions = quadratic_mde.high_distortion_pairs()
    outliers = pairs[:10]


    def plot_pairs(pairs):
        fig, axs = plt.subplots(2, pairs.shape[0], figsize=(15.0, 3.0))
        for pair_index in range(pairs.shape[0]):
            i = pairs[pair_index][0]
            j = pairs[pair_index][1]
            im_i = mnist.data[i].reshape(28, 28)
            im_j = mnist.data[j].reshape(28, 28)
            axs[0][pair_index].imshow(im_i)
            axs[0][pair_index].set_xticks([])
            axs[0][pair_index].set_yticks([])
            axs[1][pair_index].imshow(im_j)
            axs[1][pair_index].set_xticks([])
            axs[1][pair_index].set_yticks([])
        plt.tight_layout()


    plot_pairs(outliers)
    return pairs, plot_pairs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Low distortion pairs

    In comparison, the pairs with the lowest distortion look like very reasonable pairs of images.
    """)
    return


@app.cell
def _(pairs, plot_pairs):
    low_distortion_pairs = pairs[-10:]
    plot_pairs(low_distortion_pairs)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Including dissimilar pairs

    The quadratic MDE problem used a standardization constraint to enforce the embedding to spread out, and only used pairs of similar items.

    Instead of only relying on the constraint, we can also design the distortion functions so that they discourage dissimilar items from being close. We do this by including some pairs of dissimilar items, in addition to the pairs of similar items.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Form $\mathcal{E} = \mathcal{E}_\text{sim} \cup \mathcal{E}_\text{dis}$
    """)
    return


@app.cell
def _(knn_graph):
    similar_edges = knn_graph.edges

    dissimilar_edges = pymde.preprocess.dissimilar_edges(
        n_items=mnist.data.shape[0],
        num_edges=similar_edges.shape[0],
        similar_edges=similar_edges,
    )

    edges = torch.cat([similar_edges, dissimilar_edges])
    weights = torch.cat(
        [knn_graph.weights, -1.0 * torch.ones(dissimilar_edges.shape[0])]
    )

    plt.hist(weights.numpy())
    plt.gca()
    return edges, weights


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Define the distortion function $f$
    """)
    return


@app.cell
def _(weights):
    f = pymde.penalties.PushAndPull(
        weights=weights,
        attractive_penalty=pymde.penalties.Log1p,
        repulsive_penalty=pymde.penalties.Log,
    )
    return (f,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### A standardized embedding ...
    """)
    return


@app.cell
def _(edges, f, quadratic_mde):
    std_mde = pymde.MDE(
        n_items=mnist.data.shape[0],
        embedding_dim=2,
        edges=edges,
        distortion_function=f,
        constraint=pymde.Standardized(),
    )
    std_mde.embed(X=quadratic_mde.X, max_iter=400, verbose=True)
    std_mde.plot(color_by=mnist.attributes["digits"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## ... and an unconstrained embedding
    """)
    return


@app.cell
def _(edges, f, quadratic_mde):
    unconstrained_mde = pymde.MDE(
        n_items=mnist.data.shape[0],
        embedding_dim=2,
        edges=edges,
        distortion_function=f,
    )
    unconstrained_mde.embed(X=quadratic_mde.X, verbose=True)
    unconstrained_mde.plot(color_by=mnist.attributes["digits"])
    return


if __name__ == "__main__":
    app.run()
