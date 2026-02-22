# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.20.0",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pymde==0.2.3",
#     "torch",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)

with app.setup:
    import marimo as mo
    import pymde
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from matplotlib import colors

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Google Scholar

    This notebook shows how to use the function `pymde.preserve_distances` to produce embeddings of networks, in which the goal is to preserve the shortest-path distances in the network.

    It uses an academic co-authorship network collected from Google Scholar as a case study.
    """)
    return


@app.cell
def _():
    gscholar = pymde.datasets.google_scholar()
    return (gscholar,)


@app.cell
def _(gscholar):
    scholars_df = gscholar.other_data['dataframe']
    scholars_df
    return (scholars_df,)


@app.cell
def _(gscholar):
    coauthorship_graph = gscholar.data
    return (coauthorship_graph,)


@app.cell
def _(coauthorship_graph):
    mo.md(f"""
    {coauthorship_graph.n_items:,} authors
    """)
    return


@app.cell
def _(coauthorship_graph):
    mo.md(f"""
    {coauthorship_graph.n_edges:,} edges
    """)
    return


@app.cell
def _(coauthorship_graph):
    mo.md(f"""
    edge density: {100*(coauthorship_graph.n_edges / (coauthorship_graph.n_all_edges)):.2f} percent
    """)
    return


@app.cell
def _(coauthorship_graph):
    with mo.persistent_cache("distance_preserving_mde"):
        distance_preserving_mde = pymde.preserve_distances(
            data=coauthorship_graph,
            loss=pymde.losses.Absolute,
            max_distances=100000000.0,
            device=DEVICE,
            verbose=True,
        )
    return (distance_preserving_mde,)


@app.cell
def _(distance_preserving_mde):
    plt.figure(figsize=(12, 3))
    original_distances = np.sort(distance_preserving_mde.distortion_function.deviations.cpu().numpy())
    ax = plt.gca()
    plt.hist(original_distances, histtype='step', bins=np.arange(1, 11), density=True, cumulative=True)
    plt.xlim(1, 10)
    plt.xticks(np.arange(1, 11))
    plt.xlabel('graph distances')
    plt.gca()
    return


@app.cell
def _(distance_preserving_mde):
    with mo.persistent_cache("solved_scholar"):
        distance_preserving_mde.embed(verbose=True)
        solved_mde = distance_preserving_mde
    return (solved_mde,)


@app.cell
def _(solved_mde):
    solved_mde.distortions_cdf()
    plt.gca()
    return


@app.cell
def _(gscholar, solved_mde):
    solved_mde.plot(
        color_by=gscholar.attributes["coauthors"],
        color_map="viridis",
        figsize_inches=(12.0, 12.0),
        background_color="k",
    )
    fig = plt.gcf()
    return (fig,)


@app.cell
def _(coauthorship_graph, gscholar, solved_mde):
    edges = coauthorship_graph.edges
    indices = torch.randperm(edges.shape[0])[:1000]
    edges = edges[indices].cpu().numpy()

    solved_mde.plot(
        edges=edges,
        color_by=gscholar.attributes["coauthors"],
        color_map="viridis",
        figsize_inches=(12, 12),
    )
    return


@app.cell
def _(gscholar):
    legend = {
        "bio": colors.to_rgba("tab:purple"),
        "ai": colors.to_rgba("tab:red"),
        "cs": colors.to_rgba("tab:cyan"),
        "ee": colors.to_rgba("tab:green"),
        "physics": colors.to_rgba("tab:orange"),
    }
    scholar_disciplines_df = gscholar.other_data["disciplines"]
    topic_colors = [legend[code] for code in scholar_disciplines_df["topic"]]
    return scholar_disciplines_df, topic_colors


@app.cell
def _(scholar_disciplines_df, solved_mde, topic_colors):
    pymde.plot(
        solved_mde.X[scholar_disciplines_df["node_id"].values],
        colors=topic_colors,
        figsize_inches=(12, 12),
        background_color="black",
    )
    return


@app.cell
def _(fig):
    select = mo.ui.matplotlib(fig)
    select
    return (select,)


@app.cell
def _(scholars_df, select, solved_mde):
    _X = solved_mde.X.cpu()
    mask = select.value.get_mask(_X[:, 0], _X[:, 1])
    scholars_df.iloc[mask]
    return


if __name__ == "__main__":
    app.run()
