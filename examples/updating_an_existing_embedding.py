# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "pymde",
#     "matplotlib",
#     "torch",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App()

with app.setup:
    import matplotlib
    import marimo as mo
    import pymde
    import torch


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Updating embeddings

    This notebook shows how to add new points to an existing embedding, using MNIST as an example.

    The basic idea is to use an anchor constraint to pin the existing embedding in place, then embed the new points.
    """)
    return


@app.cell
def _():
    mnist = pymde.datasets.MNIST()
    return (mnist,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We'll start by embedding half of the MNIST data, using the preserve neighbors function.
    """)
    return


@app.cell
def _(mnist):
    n_train = 35000
    mnist_train = mnist.data[:n_train]
    mnist_train_labels = mnist.attributes['digits'][:n_train]

    mde = pymde.preserve_neighbors(mnist_train, constraint=pymde.Standardized(), verbose=True)
    mde.embed(verbose=True)
    mde.plot(color_by=mnist_train_labels)
    return mde, n_train


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Updating the embedding

    Next, we'll augment the above embedding with embedding vectors for the rest of the MNIST dataset.

    We start by making an **anchor constraint**. This constraint says that we should pin the first `n_train` items to the embedding vectors that we just computed.
    """)
    return


@app.cell
def _(mde, n_train):
    anchor_constraint = pymde.Anchored(
        anchors=torch.arange(n_train),
        values=mde.X,
    )
    return (anchor_constraint,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We now make a new MDE problem, passing in the entire MNIST dataset to the `preserve_neighbors` function. Crucially,
    we also provide the anchor constraint.
    """)
    return


@app.cell
def _(anchor_constraint, mnist):
    incremental_mde = pymde.preserve_neighbors(
        mnist.data,
        constraint=anchor_constraint,
        init='random',
        verbose=True)
    return (incremental_mde,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The last step is to call the embed method. This will optimize the placement of the embedding vectors of the new images, while keeping the old ones fixed in place.
    """)
    return


@app.cell
def _(incremental_mde):
    incremental_mde.embed(eps=1e-6, verbose=True)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The first `n_train` rows of the new embedding are identitical to the old embedding, as verified below.
    """)
    return


@app.cell
def _(incremental_mde, mde, n_train):
    (incremental_mde.X[:n_train] == mde.X).all()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Plot the final embedding

    Below we plot the entire embedding. Notice that it looks very similar to the first embedding. This suggests that the incremental embedding was effective.
    """)
    return


@app.cell
def _(incremental_mde, mnist):
    pymde.plot(incremental_mde.X, color_by=mnist.attributes['digits'])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can compare just the new embedding vectors with the old embedding vectors. Note that both plots are nearly identitcal.
    """)
    return


@app.cell
def _(incremental_mde, mnist, n_train):
    _ax = pymde.plot(incremental_mde.X[n_train:], color_by=mnist.attributes['digits'][n_train:])
    _ax.set_title('New embedding vectors')
    return


@app.cell
def _(incremental_mde, mnist, n_train):
    _ax = pymde.plot(incremental_mde.X[:n_train], color_by=mnist.attributes['digits'][:n_train])
    _ax.set_title('Old embedding vectors')
    return


if __name__ == "__main__":
    app.run()
