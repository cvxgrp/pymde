import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import torch


def latexify(figsize_inches=None, font_size=12):
    """Set up matplotlib's RC params for LaTeX plotting.

    This function only needs to be called once per Python session.

    Arguments
    ---------
    figsize_inches: tuple float (optional)
        width, height of figure on inches

    font_size: int
        Size of font.
    """

    usetex = matplotlib.checkdep_usetex(True)
    if not usetex:
        raise RuntimeError(
            "Matplotlib could not find a LaTeX installation on your machine."
        )

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
    fig_width, fig_height = (
        figsize_inches if figsize_inches is not None else (None, None)
    )
    if fig_width is None:
        fig_width = 3.39

    if fig_height is None:
        golden_mean = (np.sqrt(5) - 1.0) / 2.0  # aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    max_height_inches = 8.0
    if fig_height > max_height_inches:
        print(
            "warning: fig_height too large:"
            + fig_height
            + "so will reduce to"
            + max_height_inches
            + "inches."
        )
        fig_height = max_height_inches

    params = {
        "backend": "ps",
        "text.latex.preamble": "\\usepackage{gensymb}",
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "font.size": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "text.usetex": True,
        "figure.figsize": [fig_width, fig_height],
        "font.family": "serif",
    }

    matplotlib.rcParams.update(params)


def _format_axes(ax, spine_color="black", linewidth=0.7):

    for spine in ["left", "bottom", "top", "right"]:
        ax.spines[spine].set_color(spine_color)
        ax.spines[spine].set_linewidth(linewidth)

    # ax.xaxis.set_ticks_position('bottom')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction="out", color=spine_color)

    return ax


def _cdf(array, n_bins=None, figsize_inches=(8, 3), xlabel=None):
    array = np.sort(array)
    if n_bins is None:
        n_bins = min(array.size, 100000)
    step = int(array.size / n_bins)
    bins = np.arange(1, array.size + 1, step=step)
    if bins[-1] != array.size:
        bins = np.concatenate((bins, np.array([array.size])))
    heights = bins / float(array.size)

    plt.figure(figsize=figsize_inches)
    ax = plt.gca()
    ax.step(array[bins - 1], heights, color="tab:blue")

    nonzero_index = np.argmax(array > 0)
    if nonzero_index > 0 or array[0] > 0:
        # array is positive, can take log
        ax.set_xlim(left=array[nonzero_index], right=None)
        plt.xscale("log")

    if xlabel is not None:
        plt.xlabel(xlabel)
    _format_axes(ax)
    plt.tight_layout()


def _show_cdf(mde, n_bins=None, X=None, figsize_inches=(8, 3)):
    X = X if X is not None else mde.X
    distortions = mde.distortions(X).cpu().numpy()
    _cdf(
        distortions,
        n_bins=n_bins,
        figsize_inches=figsize_inches,
        xlabel="distortions",
    )


def _labels_to_codes(labels):
    unique_labels = sorted(list(set(labels)))
    label2code = {label: code for code, label in enumerate(unique_labels)}
    codes = [label2code[label] for label in labels]
    return codes, unique_labels


def _square_lim(lim_low, lim_high):
    if abs(lim_low) > abs(lim_high) and lim_low < 0:
        return (lim_low, -lim_low)
    elif abs(lim_low) <= abs(lim_high) and lim_high > 0:
        return (-lim_high, lim_high)
    else:
        return (lim_low, lim_high)


def _is_discrete(dtype):
    return any(
        [
            np.issubdtype(dtype, other)
            for other in (
                np.integer,
                np.bool_,
                np.string_,
                np.unicode_,
                np.object_,
            )
        ]
    )


def _plot_3d(
    X, color_by, cmap, colors, edges, s, background_color, figsize, lim
):
    from mpl_toolkits.mplot3d import Axes3D

    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    shadowsize = 0.03
    shadowcolor = "gainsboro"

    fig = plt.figure(figsize=figsize)
    ax = Axes3D(fig)

    x, y, z = X[:, 0], X[:, 1], X[:, 2]

    if color_by is not None:
        discrete = _is_discrete(color_by.dtype)
        if discrete:
            c, unique_labels = _labels_to_codes(color_by)
        else:
            c, unique_labels = color_by, None
    elif colors is not None:
        c = colors
        cmap = None
    else:
        c = None
        cmap = None
    im = ax.scatter(
        x,
        y,
        z,
        c=c,
        edgecolor="k",
        cmap=cmap,
        alpha=1.0,
        s=s,
        linewidth=s / 20.0,
    )

    if lim is None:
        lim_low = min(np.min(x), np.min(y), np.min(z)) * 1.1
        lim_high = max(np.max(x), np.max(y), np.max(z)) * 1.1
        lim = _square_lim(lim_low, lim_high)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_zlim(lim)

    if edges is None:
        ax.plot(
            y,
            z,
            "g+",
            zdir="x",
            zs=ax.axes.get_xlim3d()[0],
            c=shadowcolor,
            alpha=0.5,
            marker="o",
            markersize=shadowsize,
        )
        ax.plot(
            x,
            y,
            "k+",
            zdir="z",
            zs=ax.axes.get_zlim3d()[0],
            c=shadowcolor,
            alpha=0.5,
            marker="o",
            markersize=shadowsize,
        )

    if background_color is not None:
        if isinstance(background_color, str):
            bg = matplotlib.colors.to_rgba(background_color)
        else:
            bg = background_color
    elif edges is not None:
        bg = matplotlib.colors.to_rgba("k")
    else:
        bg = None

    if bg is not None:
        ax.xaxis.set_pane_color(bg)
        ax.yaxis.set_pane_color(bg)
        ax.zaxis.set_pane_color(bg)

    if edges is not None:
        if isinstance(edges, torch.Tensor):
            edges = edges.cpu().numpy()
        linewidth = 4.0 / np.log(edges.shape[0])
        for e in edges:
            xi = X[e[0]]
            xj = X[e[1]]
            ax.plot(
                [xi[0], xj[0]],
                [xi[1], xj[1]],
                [xi[2], xj[2]],
                color="white",
                alpha=0.5,
                linewidth=linewidth,
                zorder=5,
            )

    if color_by is not None:
        if discrete:
            cbar = fig.colorbar(
                im,
                boundaries=np.arange(len(set(color_by)) + 1) - 0.5,
                shrink=0.5,
                orientation="horizontal",
                anchor=(0.5, 1.8),
            )
            cbar.set_ticks(np.arange(len(unique_labels)))
            if isinstance(unique_labels[0], str):
                rotation = 90
            else:
                rotation = 0
            cbar.ax.set_xticklabels(unique_labels, rotation=rotation)
        else:
            cbar = fig.colorbar(
                im,
                shrink=0.5,
                orientation="horizontal",
                anchor=(0.5, 1.8),
            )

    return ax


def _plot(
    X=None,
    color_by=None,
    edges=None,
    s=1.0,
    figsize=(6.4, 6.0),
    colors=None,
    cmap="Spectral",
    lim=None,
    background_color=None,
    title=None,
    tight=True,
    savepath=None,
):
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    else:
        X = np.asarray(X)

    if isinstance(color_by, torch.Tensor):
        color_by = color_by.cpu().numpy()
    elif color_by is not None:
        color_by = np.asarray(color_by)

    if isinstance(edges, torch.Tensor):
        edges = edges.cpu().numpy()
    elif edges is not None:
        edges = np.asarray(edges)

    if isinstance(colors, torch.Tensor):
        colors = colors.cpu().numpy()
    elif colors is not None:
        colors = np.asarray(colors)

    X = X[:, :3]
    if X.shape[1] == 3:
        ax = _plot_3d(
            X,
            lim=lim,
            color_by=color_by,
            cmap=cmap,
            colors=colors,
            edges=edges,
            s=s,
            background_color=background_color,
            figsize=figsize,
        )
    else:
        if X.shape[1] == 1:
            X = np.hstack([X, np.zeros((X.shape[0], 1))])

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if background_color is not None:
            ax.set_facecolor(background_color)

        if lim is not None:
            ax.set_xlim(lim)
            ax.set_ylim(lim)
        else:
            lim_low = min(np.min(X[:, 0]), np.min(X[:, 1])) * 1.1
            lim_high = max(np.max(X[:, 0]), np.max(X[:, 1])) * 1.1
            lim = _square_lim(lim_low, lim_high)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal", adjustable="box")

        # scatterplot of embedding
        if color_by is not None:
            discrete = _is_discrete(color_by.dtype)
            if discrete:
                codes, unique_labels = _labels_to_codes(color_by)
            else:
                codes = color_by.data
                unique_labels = None

            # edgecolor=[] prevents small points from having no fill
            im = ax.scatter(
                X[:, 0],
                X[:, 1],
                s=s,
                c=codes,
                edgecolor=[],
                cmap=cmap,
                alpha=1.0,
            )
        elif colors is not None:
            im = ax.scatter(
                X[:, 0],
                X[:, 1],
                s=s,
                c=colors,
                edgecolor=[],
                alpha=1.0,
            )
        else:
            ax.scatter(X[:, 0], X[:, 1], s=s, alpha=1.0)

        if edges is not None:
            linewidth = 6.0 / np.log(edges.shape[0])
            for e in edges:
                ax.plot(
                    [X[e[0], 0], X[e[1], 0]],
                    [X[e[0], 1], X[e[1], 1]],
                    linestyle="-",
                    marker="o",
                    markersize=s,
                    color="white",
                    alpha=0.5,
                    linewidth=linewidth,
                )
            ax.set_facecolor("k")

        if title is not None:
            ax.set_title(title)

        _format_axes(ax)
        if tight:
            plt.tight_layout()

        if color_by is not None:
            aspect = 30
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3.%", pad=0.05)
            if discrete:
                n_labels = len(unique_labels)
                cbar = fig.colorbar(
                    im,
                    ax=ax,
                    cax=cax,
                    boundaries=np.arange(n_labels + 1) - 0.5,
                    aspect=aspect,
                )
                cbar.set_ticks(np.arange(n_labels))
                cbar.ax.set_yticklabels(unique_labels)
            else:
                cbar = fig.colorbar(
                    im,
                    ax=ax,
                    cax=cax,
                    aspect=aspect,
                )

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")

    return ax


def plot(
    X,
    color_by=None,
    color_map="Spectral",
    colors=None,
    edges=None,
    axis_limits=None,
    background_color=None,
    marker_size=1.0,
    figsize_inches=(8.0, 8.0),
    savepath=None,
):
    """Plot an embedding, in one, two, or three dimensions.

    This function plots embeddings. The input embedding's dimension should
    be at most 3.

    The embedding is visualized as a scatter plot. The points can
    optionally be colored according to categorical or continuous values,
    or according to a pre-defined sequence of colors. Additionally,
    edges can optionally be superimposed.

    Arguments
    ---------
    X: array-like
        The embedding to plot, of shape ``(n_items, embedding_dim)``. The
        second dimension should be 1, 2, or 3.
    color_by: array-like, optional
        A sequence of values, one for each item, which should be
        used to color each embedding vector. These values may either
        be categorical or continuous. For example, if ``n_items`` is 4,

        .. code:: python3

            np.ndarray(['dog', 'cat', 'zebra', 'cat'])
            np.ndarray([0, 1, 1, 2]
            np.ndarray([0.1, 0.5, 0.31, 0.99]

        are all acceptable. The first two are treated as categorical,
        the third is continuous. A finite number of colors is used
        when the values are categorical, while a spectrum of colors is
        used when the values are continuous.
    color_map: str or matplotlib colormap instance
        Color map to use when resolving ``color_by`` to colors; ignored
        when ``color_by`` is None.
    colors: array-like, optional
        A sequence of colors, one for each item, specifying the exact
        color each item should be colored. Each row must represent
        an RGBA value.

        Only one of ``color_by`` and ``colors`` should be non-None.
    edges: array-like, optional
        List of edges to superimpose over the scatter plot, shape ``(any, 2)``
    axis_limits: tuple, optional
        tuple ``(limit_low, limit_high)`` of axis limits, applied to both
        the x and y axis.
    background_color: str, optional
        color of background
    marker_size: float, optional
        size of each point in the scatter plot
    figsize_inches: tuple
        size of figures in inches: ``(width_inches, height_inches)``
    savepath: str, optional
        path to save the plot.

    Returns
    -------
    matplotlib.Axes:
        Axis on which the embedding is plotted.
    """
    if color_by is not None and colors is not None:
        raise ValueError("Only one of 'color_by` and `colors` can be non-None")

    ax = _plot(
        X=X,
        color_by=color_by,
        cmap=color_map,
        colors=colors,
        edges=edges,
        lim=axis_limits,
        background_color=background_color,
        s=marker_size,
        figsize=figsize_inches,
    )

    if savepath is not None:
        plt.savefig(savepath)
    return ax


def _plot_gif_frame(
    snapshot,
    color_by,
    cmap,
    colors,
    edges,
    lim,
    background_color,
    figsize_inches,
    outdir,
    title,
    i,
    s,
):
    _plot(
        X=snapshot,
        color_by=color_by,
        cmap=cmap,
        edges=edges,
        title=title,
        lim=lim,
        background_color=background_color,
        tight=False,
        s=s,
        figsize=figsize_inches,
    )
    plt.savefig(outdir + "/{:04d}.png".format(i), dpi=144)
    plt.close("all")
