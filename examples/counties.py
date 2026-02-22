# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.20.0",
#     "matplotlib",
#     "pandas",
#     "pymde==0.2.3",
#     "torch",
# ]
# ///

import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")

with app.setup:
    import matplotlib.pyplot as plt
    import pymde

    import marimo as mo

    pymde.seed(0)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Counties

    This notebook accompanies chapter 10 of the monograph [Minimum-Distortion Embedding](https://web.stanford.edu/~boyd/papers/min_dist_emb.html).

    In this eample notebook, we'll use PyMDE to embed and visualize 3,220 US counties, described by their demographic data (collected between 2013-2017 by an ACS longitudinal survey).

    We'll then color each county by the fraction of voters who voted for a democratic candidate in the 2016 presidential election. Interestingly, the embedding vaguely resembles a map of the US, though no geographic data was used to compute the embedding.
    """)
    return


@app.cell
def _():
    dataset = pymde.datasets.counties()
    return (dataset,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below is the raw dataframe, which was preprocessed using the code in scripts/preprocess_counties_data.py.

    The preprocessed data is stored in dataset.data
    """)
    return


@app.cell
def _(dataset):
    dataset.county_dataframe
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We now make a neighbor-preserving embedding, to explore the local relationships in the data.
    """)
    return


@app.cell
def _(dataset):
    dataset
    return


@app.cell
def _(dataset):
    with mo.persistent_cache("counties"):
        mde = pymde.preserve_neighbors(data=dataset.data, verbose=True)
        embedding = mde.embed()
    return (embedding,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Finally we visualize the embedding, rotating it so that it vaguely resembles a map of the US.

    Note that counties that voted Republican tend to cluster together, as do counties that voted Democratic.
    """)
    return


@app.cell
def _(dataset, embedding):
    # Rotate the embedding by some amount of degrees
    rotated_embedding = pymde.rotate(embedding, 150)
    ax = pymde.plot(
        rotated_embedding,
        color_by=dataset.attributes["democratic_fraction"],
        color_map="RdBu",
        marker_size=10,
    )
    plt.tight_layout()
    ax
    return


if __name__ == "__main__":
    app.run()
