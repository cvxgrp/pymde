# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "pandas",
#     "pymde==0.3.0",
# ]
# ///

import marimo

__generated_with = "0.19.10"
app = marimo.App()

with app.setup:
    import marimo as mo
    import pymde


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Population genetics

    This notebook accompanies chapter 11 of the monograph [Minimum-Distortion Embedding](https://web.stanford.edu/~boyd/papers/min_dist_emb.html).

    In this eample notebook, we'll embed and visualize genomic data for 1,387 people thought to be of European ancestry, along with 154 synthetic (corrupted) points. We'll see that the embedding vaguely resembles the map of the Europe --- and that the synthetic points are isolated in their own cluster --- suggesting that the embedding is robust to noise, and that the genetic data encodes geographical data.
    """)
    return


@app.cell
def _():
    dataset = pymde.datasets.population_genetics()
    return (dataset,)


@app.cell
def _(dataset):
    mde = pymde.preserve_neighbors(
        data=dataset.corrupted_data,
        attractive_penalty=pymde.penalties.Huber,
        repulsive_penalty=pymde.penalties.Log,
        verbose=True
    )
    embedding = mde.embed()
    return (embedding,)


@app.cell
def _(dataset, embedding):
    pymde.plot(embedding, colors=dataset.attributes['corrupted_colors'], marker_size=10)
    return


if __name__ == "__main__":
    app.run()
