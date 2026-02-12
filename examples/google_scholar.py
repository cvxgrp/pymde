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
    from matplotlib import colors
    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import pymde
    import torch


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
    return


@app.cell
def _(gscholar):
    coauthorship_graph = gscholar.data
    return (coauthorship_graph,)


@app.cell
def _(coauthorship_graph):
    f'{coauthorship_graph.n_items:,} authors'
    return


@app.cell
def _(coauthorship_graph):
    f'{coauthorship_graph.n_edges:,} edges'
    return


@app.cell
def _(coauthorship_graph):
    print(f'edge density: {100*(coauthorship_graph.n_edges / (coauthorship_graph.n_all_edges)):.2f} percent')
    return


@app.cell
def _(coauthorship_graph):
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mde = pymde.preserve_distances(data=coauthorship_graph, loss=pymde.losses.Absolute, max_distances=100000000.0, device=_device, verbose=True)
    return (mde,)


@app.cell
def _(mde):
    plt.figure(figsize=(12, 3))
    original_distances = np.sort(mde.distortion_function.deviations.cpu().numpy())
    ax = plt.gca()
    plt.hist(original_distances, histtype='step', bins=np.arange(1, 11), density=True, cumulative=True)
    plt.xlim(1, 10)
    plt.xticks(np.arange(1, 11))
    plt.xlabel('graph distances')
    plt.gca()
    return


@app.cell
def _(mde):
    mde.embed(verbose=True)
    return


@app.cell
def _(mde):
    mde.distortions_cdf()
    return


@app.cell
def _(gscholar, mde):
    mde.plot(color_by=gscholar.attributes['coauthors'], color_map='viridis',
             figsize_inches=(12., 12.), background_color='k')
    return


@app.cell
def _(coauthorship_graph, gscholar, mde):
    edges = coauthorship_graph.edges
    indices = torch.randperm(edges.shape[0])[:1000]
    edges = edges[indices].cpu().numpy()

    mde.plot(edges=edges, color_by=gscholar.attributes['coauthors'], color_map='viridis', figsize_inches=(12, 12))
    return


@app.cell
def _(gscholar):
    legend = {
        'bio': colors.to_rgba('tab:purple'),
        'ai': colors.to_rgba('tab:red'),
        'cs': colors.to_rgba('tab:cyan'),
        'ee': colors.to_rgba('tab:green'),
        'physics': colors.to_rgba('tab:orange')
    }
    scholar_disciplines_df = gscholar.other_data['disciplines']
    topic_colors = [legend[code] for code in scholar_disciplines_df['topic']]
    return scholar_disciplines_df, topic_colors


@app.cell
def _(mde, scholar_disciplines_df, topic_colors):
    pymde.plot(mde.X[scholar_disciplines_df['node_id'].values], colors=topic_colors,
               figsize_inches=(12, 12), background_color='black')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A neighbor-preserving embedding
    """)
    return


@app.cell
def _(coauthorship_graph, scholar_disciplines_df, topic_colors):
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mde_1 = pymde.preserve_neighbors(data=coauthorship_graph, device=_device, verbose=True)
    mde_1.embed(verbose=True)
    pymde.plot(mde_1.X[scholar_disciplines_df['node_id'].values], colors=topic_colors, figsize_inches=(12, 12), background_color='black')
    return


if __name__ == "__main__":
    app.run()
