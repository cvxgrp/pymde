# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "pandas",
#     "pymde==0.2.3",
#     "torch==2.10.0",
#     "wigglystuff==0.2.27",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

with app.setup:
    import matplotlib.pyplot as plt
    import pymde
    import torch
    from wigglystuff import ChartSelect

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
    chart = mo.ui.anywidget(ChartSelect(ax.figure))
    chart
    return chart, rotated_embedding


@app.cell
def _(chart, rotated_embedding):
    selected_indices = chart.get_indices(rotated_embedding[:, 0], rotated_embedding[:, 1])
    return (selected_indices,)


@app.cell(hide_code=True)
def _(dataset, selected_indices):
    mo.stop(not selected_indices.size)

    selected_counties = dataset.county_dataframe.iloc[selected_indices]
    table = mo.ui.table(selected_counties)

    mo.output.replace(
        mo.md(
            f"""
        **You've selected {len(selected_indices)} counties.**

        {table}
        """
        )
    )
    return


if __name__ == "__main__":
    app.run()
