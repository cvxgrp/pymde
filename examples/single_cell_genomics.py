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
    import marimo as mo
    import pymde
    import torch


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Single-Cell Genomics

    In this example notebook, we show how to use PyMDE to compute an embedding of some single-cell genomics data.

    We'll embed roughly 44,000 PBMCs, specified by their mRNA transcriptomes, into 3 dimensions. These PBMCs were collected from some patients who were healthy, and others who had severe COVID-19 infections.

    After embedding, we'll plot the embedding and analyze it by coloring the vectors by cell type and patient health status. We'll also analyze where the embedding succeeded, and where it failed.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The original data
    The data is from [A single-cell atlas of the peripheral immune response in patients with severe COVID-19](https://www.nature.com/articles/s41591-020-0944-y), by Wilk, et al (2020).

    The original data is available at https://cellxgene.cziscience.com/
    """)
    return


@app.cell
def _():
    scrna_wilk = pymde.datasets.covid19_scrna_wilk()
    return (scrna_wilk,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The data is a matrix in which each row is a PCA embedding of a cell's mRNA transcriptome.
    """)
    return


@app.cell
def _(scrna_wilk):
    scrna_wilk.data.shape
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can visualize the top 3 components of the PCA embedding.
    """)
    return


@app.cell
def _(scrna_wilk):
    pymde.plot(scrna_wilk.data, color_by=scrna_wilk.attributes['cell_type'], color_map='tab20')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Embedding

    We'll compute an embedding via `pymde.preserve_neighbors`. We use this function, instead of `pymde.preserve_distances`, because we're interested more in the local structure of the data than the global.
    The exact distance between transcriptomes does not matter much, and might not even be that reliable.

    In the below cell, we've chosen to compute a standardized embedding. Standardized embeddings are generally reasonable: the constraint prevents the vectors from spreading out too much, and enforces the feature columns of the embedding to be uncorrelated.

    Feel free to experiment with the keyword arguments of the `preserve_neighbors` function. For example, you can omit the constraint (replace it with `None`), vary the `repulsive_fraction` or the number of neighbors `n_neighbors`, try a random initialization (`init='random'`), or modify the attractive and repulsive penalties.

    For a full list of the configurable options, type `pymde.preserve_neighbors?` into a code cell, and execute it.
    """)
    return


@app.cell
def _(scrna_wilk):
    mde = pymde.preserve_neighbors(
        scrna_wilk.data,
        embedding_dim=3,
        constraint=pymde.Standardized(),
        repulsive_fraction=1.0,
        verbose=True,
    )
    embedding = mde.embed(verbose=True)
    return embedding, mde


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Next, we plot the embedding.

    Because the embedding is in 3 dimensions, some viewing angles look better than others. You can use `pymde.rotate` to rotate the embedding until you find a good angle.

    (Alternatively, use external libraries like Polyscope to explore 3D embeddings interactively.)
    """)
    return


@app.cell
def _(embedding, scrna_wilk):
    angles = torch.tensor([50., 167., 225.])
    rotated_embedding = pymde.rotate(embedding, angles)
    pymde.plot(rotated_embedding, color_by=scrna_wilk.attributes['cell_type'], color_map='tab20',
               figsize_inches=(12, 12))
    return (rotated_embedding,)


@app.cell
def _(rotated_embedding, scrna_wilk):
    pymde.plot(rotated_embedding, color_by=scrna_wilk.attributes['health_status'], color_map='coolwarm',
               figsize_inches=(12, 12))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below, we plot a random sample of edges between pairs of similar cells. You'll see that the edges trace out the clusters in the embedding, and that different clusters are loosely connected to each other.
    """)
    return


@app.cell
def _(mde, rotated_embedding, scrna_wilk):
    similar_cells = mde.edges[mde.distortion_function.pos_idx]
    sampled_edges = similar_cells[torch.randperm(similar_cells.shape[0])[:10000]]
    pymde.plot(rotated_embedding, color_by=scrna_wilk.attributes['cell_type'], color_map='tab20', edges=sampled_edges, figsize_inches=(12, 12))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In the next cell, we plot the 500 pairs of cells that experienced the largest distortion. These represent pairs of cells that were difficult to embed.

    In general, it can be instructive to manually inspect the original data corresponding to the most highly distorted pairs. You might find that the pairs don't actually represent similar items (cells), or maybe that the items are in some way outliers, anomalous, or corrupted.
    """)
    return


@app.cell
def _(mde, rotated_embedding, scrna_wilk):
    pairs, _ = mde.high_distortion_pairs()
    most_distorted_pairs = pairs[:500]
    pymde.plot(rotated_embedding, color_by=scrna_wilk.attributes['cell_type'], color_map='tab20',
               edges=most_distorted_pairs, figsize_inches=(12, 12))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Finally, we plot the CDF of distortions below. You can see that most cells experience little distortions, but a few cells --- roughly 10 or 20 percent --- have much higher distortion.
    """)
    return


@app.cell
def _(mde):
    mde.distortions_cdf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In this notebook, we computed a neighbor-preserving embedding, since we were more interested in the local structure than the global structure of the data.

    If you are curious what a distance-preserving embedding might look like, you can compute one using the following code:

    ```python3

    mde = pymde.preserve_distances(
        scrna_wilk.data,
        embedding_dim=3,
        verbose=True,
        max_distances=1e7,
    )
    embedding = mde.embed(verbose=True)
    ```

    You'll find that it looks similar to the original PCA embedding.
    """)
    return


if __name__ == "__main__":
    app.run()
