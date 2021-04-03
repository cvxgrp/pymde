"""Minimum-Distortion Embedding

This module defines the MDE class, which represents an MDE problem.
"""


import copy
import glob
import logging
import sys
import tempfile
import typing as tp

import torch

from pymde.average_distortion import _average_distortion
from pymde.average_distortion import _gather_indices, _norm
from pymde import constraints
from pymde.functions.function import StochasticFunction
from pymde.util import _canonical_device, _module_device
from pymde import optim


LOGGER = logging.getLogger("__pymde__")
LOGGER.propagate = False
LOGGER.setLevel(logging.INFO)
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setLevel(logging.INFO)
_formatter = logging.Formatter(
    fmt="%(asctime)s: %(message)s", datefmt="%b %d %I:%M:%S %p"
)
_stream_handler.setFormatter(_formatter)
LOGGER.addHandler(_stream_handler)


class MDE(torch.nn.Module):
    """An MDE problem.

    An MDE instance represents a specific MDE problem, specified by
    the number of items, the embedding dimension, a list of edges,
    the vector distortion function, and possibly a constraint.

    Attributes
    ----------
    n_items: int
        The number of items
    embedding_dim: int
        The embedding dimension
    edges: torch.Tensor
        The list of edges.
    distortion_function: Callable
        The vector distorton function.
    constraint: pymde.constraints.Constraint
        The constraint imposed (or None)
    device
        The device on which to compute the embedding and store data
        (like 'cpu' or 'cuda')
    solve_stats
        Summary statistics about the embedding, populated after calling
        the ``embed`` method.
    """
    def __init__(
        self,
        n_items: int,
        embedding_dim: int,
        edges: torch.Tensor,
        distortion_function: tp.Union[tp.Callable, StochasticFunction],
        constraint: tp.Optional[constraints.Constraint] = None,
        device: tp.Optional[str] = None,
    ):
        """Constructs an MDE problem.

        Arguments
        ---------
        n_items: int
            Number of things being embedded.
        embedding_dim: int
            Embedding dimension.
        edges: torch.Tensor(shape=(num_edges, 2), dtype=torch.int)
            Tensor, where each row is an edge (i, j) between two items;
            each edge should satisfy 0 <= i < j < n_items. In particular
            self-edges are not allowed.
        distortion_function: Callable or pymde.functions.StochasticFunction
            The vectorized distortion function, typically an instance
            of a class from ``pymde.penalties`` or ``pymde.losses`` however,
            this can be any Python callable that maps a torch.Tensor
            of embedding distances to a torch.Tensor of distortions.
        constraint: pymde.constraints.Constraint, optional
            A Constraint object, such as ``pymde.Standardized()``
            Defaults to an unconstrained (centered) embedding.
        device: str, optional
            Name of device on which to store tensors/compute embedding,
            such as 'cpu' or 'cuda' for GPU. Default infers device from
            ``edges`` and ``distortion_function``
        """
        super(MDE, self).__init__()
        if device is None:
            if (
                isinstance(edges, torch.Tensor)
                and isinstance(distortion_function, torch.nn.Module)
            ) and (
                str(edges.device) == str(_module_device(distortion_function))
            ):
                device = edges.device
            else:
                device = "cpu"
        self.device = _canonical_device(device)

        if not isinstance(n_items, torch.Tensor):
            n_items = torch.tensor(n_items, device=self.device)
        elif str(n_items.device) != str(self.device):
            n_items = n_items.to(self.device)
        self.register_buffer("n_items", n_items)

        if not isinstance(embedding_dim, torch.Tensor):
            embedding_dim = torch.tensor(embedding_dim, device=self.device)
        elif str(embedding_dim.device) != str(self.device):
            embedding_dim = embedding_dim.to(self.device)
        self.register_buffer("embedding_dim", embedding_dim)

        if edges is None:
            if not isinstance(distortion_function, StochasticFunction):
                raise ValueError(
                    "edges can only be None when using a stochastic function."
                )
            p = distortion_function.p
        else:
            if not isinstance(edges, torch.Tensor):
                edges = torch.tensor(
                    edges, dtype=torch.int64, device=self.device
                )

            if (edges[:, 0] == edges[:, 1]).any():
                offending = torch.where(edges[:, 0] == edges[:, 1])[0]
                raise ValueError(
                    "The edge list must not contain self edges; the "
                    "following rows were found to be self edges: ",
                    offending.cpu().numpy(),
                )

            if str(edges.device) != str(self.device):
                LOGGER.warning(
                    "edges.device (%s) "
                    "does not match requested device (%s); copying edges to "
                    "requested device." % (edges.device, device)
                )
                edges = edges.to(self.device)
            p = torch.tensor(edges.shape[0], device=self.device)

        complete_graph_edges = n_items * (n_items - 1) // 2
        if p is not None and p > complete_graph_edges:
            raise ValueError(
                "Your graph has more than (n_items choose 2) edges."
                "(p: {0}, n_items choose 2: {1})".format(
                    p, complete_graph_edges
                )
            )

        self.register_buffer("edges", edges)
        self.register_buffer("p", p)
        self.register_buffer("_complete_graph_edges", complete_graph_edges)

        if edges is not None:
            self.register_buffer(
                "_lhs", _gather_indices(edges[:, 0], self.embedding_dim)
            )
            self.register_buffer(
                "_rhs", _gather_indices(edges[:, 1], self.embedding_dim)
            )

        if isinstance(distortion_function, torch.nn.Module):
            f_device = _module_device(distortion_function)
            if f_device is None or str(f_device) != str(self.device):
                LOGGER.warning(
                    "distortion_function device (%s) "
                    "does not match requested device (%s); making a copy of "
                    "distortion_function" % (str(f_device), device)
                )
                distortion_function = copy.deepcopy(distortion_function)
                distortion_function.to(self.device)
        self.distortion_function = distortion_function

        if constraint is None:
            constraint = constraints.Centered()
        self.constraint = constraint

        self.register_buffer("X", None)
        self.register_buffer("_X_init", None)

        self.solve_stats = None
        self.value = None
        self.residual_norm = None

    def to(self, device):
        """Move MDE instance to another device."""
        super(MDE, self).to(device)
        self.device = _canonical_device(device)

    def __str__(self):
        func_name = (
            self.distortion_function.__name__
            if hasattr(self.distortion_function, "__name__")
            else type(self.distortion_function).__name__
        )
        if self.p is None:
            return (
                "Stochastic MDE problem:\n"
                "\tn (number of items) {0}\n"
                "\tm (embedding dimension) {1}\n"
                "\t{2} distortion functions\n"
                "\tconstraint {3}\n"
                "\tdevice {4}".format(
                    self.n_items.item(),
                    self.embedding_dim.item(),
                    func_name,
                    self.constraint.name(),
                    self.device,
                )
            )

        return (
            "MDE problem:\n"
            "\tn (number of items) {0}\n"
            "\tm (embedding dimension) {1}\n"
            "\tp (number of edges) {2}\n"
            "\tfraction of total edges {3:.1e}\n"
            "\t{4} distortion functions\n"
            "\tconstraint {5}\n"
            "\tdevice {6}".format(
                self.n_items.item(),
                self.embedding_dim.item(),
                self.p,
                (float(self.p) / self._complete_graph_edges).item(),
                func_name,
                self.constraint.name(),
                self.device,
            )
        )

    def _repr_pretty_(self, p, cycle):
        del cycle
        text = self.__str__()
        p.text(text)

    def differences(self, X):
        """Compute ``X[i] - X[j]`` for each row (i, j) in edges."""
        lhs = X.gather(0, self._lhs)
        rhs = X.gather(0, self._rhs)
        return lhs.sub_(rhs)

    def distances(self, X=None):
        """Compute embedding distances.

        This function computes the embedding distances corresponding to
        the list of edges (``self.edges``)

        If ``X`` is None, computes distances for the embedding computed
        by the last call to ``embed``

        Arguments
        ---------
        X: torch.Tensor(shape=(n_items, 2)), optional
            Embedding.

        Returns
        -------
        torch.Tensor(shape=(n_edges,))
            Vector of embedding distances. The k-th entry is the distance
            corresponding to the k-th edge (``self.edges[k]``
        """
        if X is None:
            X = self.X
        if X is None:
            raise ValueError(
                "Call this function after running the `embed` method, or "
                "provide a value for the embedding argument `X`"
            )
        return _norm(self.differences(X))

    def distortions(self, X=None):
        """Compute distortions

        This function computes the distortions for an embedding.

        If ``X`` is None, computes distortions for the embedding computed
        by the last call to ``embed``

        Arguments
        ---------
        X: torch.Tensor(shape=(n_items, 2)), optional
            Embedding.

        Returns
        -------
        torch.Tensor(shape=(n_edges,))
            Vector of distortions. The k-th entry is the distortion
            for the k-th edge (``self.edges[k]``
        """
        if X is None:
            X = self.X
        if X is None:
            raise ValueError(
                "Call this function after running the `embed` method, or "
                "provide a value for the embedding argument `X`"
            )
        return self.distortion_function(self.distances(X))

    def average_distortion(self, X=None):
        """Compute average distortion.

        This method computes the average distortion of an embedding.

        If ``X`` is None, this method computes the average distortion for the
        embedding computed by the last call to ``embed``.

        Arguments
        ---------
        X: torch.Tensor(shape=(n_items, 2)), optional
            Embedding.

        Returns
        -------
        scalar (float)
            The average distortion.
        """
        if X is None:
            X = self.X
        if X is None:
            raise ValueError(
                "Call this function after running the `embed` method, or "
                "provide a value for the embedding argument `X`"
            )
        return _average_distortion(
            X, self.distortion_function, self._lhs, self._rhs
        )

    def high_distortion_pairs(self, X=None):
        """Compute distortions, sorted from high to low.

        Computes the distortions for an embedding, sorting them from high
        to low. Returns the edges and distortions in the sorted order.

        This function can be used to debug an embedding or search for outliers
        in the data. In particular it can be instructive to manually
        examine the items which incurred the highest distortion.

        For example:

        .. code:: python3

            edges, distortions = mde.high_distortion_pairs()
            maybe_outliers = edges[:10]

        You can then examine the items corresponding to the edges in
        ``maybe_outliers``.

        Arguments
        ---------
        X: torch.Tensor(shape=(n_items, 2)), optional
            Embedding.

        Returns
        -------
        torch.Tensor(shape=(n_edges, 2))
            edges, sorted from highest distortion to lowest

        torch.Tensor(shape=(n_edges,))
            distortions, sorted from highest to lowest
        """
        if X is None:
            X = self.X

        if X is None:
            raise ValueError(
                "Call this function after running the `embed` method, or "
                "provide a value for the embedding argument `X`"
            )
        distortions = self.distortions(X)
        # everything after `argsort()` just reverses the array
        indices = distortions.argsort()[:, None].flipud().flatten()
        distortions_high_to_low = distortions[indices]
        sorted_pairs = self.edges[indices]
        return sorted_pairs, distortions_high_to_low

    def embed(
        self,
        X=None,
        eps=1e-5,
        max_iter=300,
        memory_size=10,
        verbose=False,
        print_every=None,
        snapshot_every=None,
    ):
        """Compute an embedding.

        This method stores the embedding in the ``X`` attribute of the problem
        instance (``mde.X``). Summary statistics related to the fitting
        process are stored in ``solve_stats`` (``mde.solve_stats``).

        All arguments have sensible default values, so in most cases,
        it suffices to just type ``mde.embed()`` or ``mde.embed(verbose=True)``

        Arguments
        ---------
        X: torch.Tensor, optional
            Initial iterate, of shape ``(n_items, embedding_dim)``. When None,
            the initial iterate is chosen randomly (and projected onto the
            constraints); otherwise, the initial iterate should satisfy
            the constraints.
        eps: float
            Residual norm threshold; quit when the residual norm
            is smaller than ``eps``.
        max_iter: int
            Maximum number of iterations.
        memory_size: int
            The quasi-Newton memory. Larger values may lead to more stable
            behavior, but will increase the amount of time each iteration
            takes.
        verbose: bool
            Whether to print verbose output.
        print_every: int, optional
            Print verbose output every ``print_every`` iterations.
        snapshot_every: int, optional
            Snapshot embedding every ``snapshot_every`` iterations;
            snapshots saved as CPU tensors to ``self.solve_stats.snapshots``.
            If you want to generate an animation with the ``play`` method after
            embedding, set ``snapshot_every`` to a positive integer
            (like 1 or 5).

        Returns
        -------
        torch.Tensor
            The embedding, of shape ``(n_items, embedding_dim)``.
        """
        if isinstance(self.distortion_function, StochasticFunction):
            device = (self.distortion_function.device,)
        else:
            device = self.device

        if X is None and self._X_init is not None:
            X = self._X_init.detach().clone()
        elif X is None:
            X = self.constraint.initialization(
                self.n_items, self.embedding_dim, device
            )
        else:
            X = X.detach().clone()

        if X.device != self.device:
            LOGGER.warning(
                f"The initial iterate's device ({X.device}) does not match "
                f"the requested device ({device}). Copying the iterate to "
                f"{device}."
            )
            X = X.to(device)

        if max_iter < 0:
            raise ValueError("`max_iter` must be greater than 0")
        if memory_size <= 0:
            raise ValueError("`memory_size` must be greater than 0")
        requires_grad = X.requires_grad

        if verbose:
            LOGGER.info(
                (
                    f"Fitting a {self.constraint.name()} embedding into "
                    f"R^{int(self.embedding_dim)}, for a graph with "
                    f"{int(self.n_items)} items and {int(self.p)} edges."
                )
            )
            LOGGER.info(
                f"`embed` method parameters: eps={eps:.1e}, "
                f"max_iter={max_iter}, memory_size={memory_size}"
            )

        if print_every is None:
            print_every = max(1, max_iter // 10)

        if isinstance(self.distortion_function, StochasticFunction):
            X_star, solve_stats = optim.spi(
                batch_size=self.distortion_function.batch_size,
                stochastic_function=self.distortion_function,
                X=X,
                constraint=self.constraint,
                eps=eps,
                max_iter=max_iter,
                memory_size=memory_size,
                use_line_search=True,
                verbose=verbose,
                print_every=print_every,
                snapshot_every=snapshot_every,
                logger=LOGGER,
            )
        else:
            X_star, solve_stats = optim.lbfgs(
                X=X,
                constraint=self.constraint,
                objective_fn=self.average_distortion,
                eps=eps,
                max_iter=max_iter,
                memory_size=memory_size,
                use_line_search=True,
                use_cached_loss=True,
                verbose=verbose,
                print_every=print_every,
                snapshot_every=snapshot_every,
                logger=LOGGER,
            )

        self.X = X_star
        self.solve_stats = solve_stats
        self.value = solve_stats.average_distortions[-1]
        self.residual_norm = solve_stats.residual_norms[-1]

        if verbose:
            LOGGER.info(
                f"Finished fitting in {solve_stats.solve_time:.3f} seconds "
                f"and {solve_stats.iterations} iterations."
            )
            LOGGER.info(
                f"average distortion {self.value:.3g} | "
                f"residual norm {self.residual_norm:.1e}"
            )
        X.requires_grad_(requires_grad)
        return self.X

    forward = embed

    def plot(
        self,
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

        This method plots embeddings, with embedding dimension at most 3.

        The embedding is visualized as a scatter plot. The points can
        optionally be colored according to categorical or continuous values,
        or according to a pre-defined sequence of colors. Additionally,
        edges can optionally be superimposed.

        The ``embed`` method must be called sometime before calling this method.

        Arguments
        ---------
        color_by: np.ndarray(shape=mde.n_items), optional
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
        colors: np.ndarray(shape=(mde.n_items, 4)), optional
            A sequence of colors, one for each item, specifying the exact
            color each item should be colored. Each row must represent
            an RGBA value.

            Only one of ``color_by`` and ``colors`` should be non-None.
        edges: {torch.Tensor/np.ndarray}(shape=(any, 2)), optional
            List of edges to superimpose over the scatter plot.
        axis_limits: tuple(length=2), optional
            tuple (limit_low, limit_high) of axis limits, applied to both
            the x and y axis.
        background_color: str, optional
            color of background
        marker_size: float, optional
            size of each point in the scatter plot
        figsize_inches: tuple(length=2)
            size of figures in inches: (width_inches, height_inches)
        savepath: str, optional
            path to save the plot.

        Returns
        -------
        matplotlib axis:
            Axis on which the embedding is plotted.
        """
        from pymde import experiment_utils

        if self.X is None:
            raise ValueError("The `embed` method must be called before `show`")

        if color_by is not None and colors is not None:
            raise ValueError(
                "Only one of `color_by` and `colors` "
                "should be provided, not both."
            )

        return experiment_utils.plot(
            X=self.X,
            color_by=color_by,
            color_map=color_map,
            colors=colors,
            edges=edges,
            axis_limits=axis_limits,
            background_color=background_color,
            figsize_inches=figsize_inches,
            marker_size=marker_size,
            savepath=savepath,
        )

    def distortions_cdf(self, figsize_inches=(8, 3)):
        """Plot a cumulative distribution function of the distortions.

        The ``embed`` method must be called sometime before calling this method.
        """
        from pymde import experiment_utils

        if self.X is None:
            raise ValueError(
                "This method can only be called after calling `embed`"
            )
        experiment_utils._show_cdf(self, figsize_inches=figsize_inches)

    def play(
        self,
        color_by=None,
        color_map="Spectral",
        colors=None,
        edges=None,
        axis_limits=None,
        background_color=None,
        marker_size=1.0,
        figsize_inches=(8.0, 8.0),
        fps=None,
        tmpdir=None,
        savepath=None,
    ):
        """Create a movie visualizing how the embedding was formed.

        This method creates a GIF showing how the embedding was formed, starting
        with the initial iterate and ending with the final embedding. In other
        words it visualizes how the embedding algorithm (the ``embed`` method)
        updated the embedding's state over time. (The embedding dimension should
        be at most 3.)

        In each frame, the embedding is visualized as a scatter plot. The points
        can optionally be colored according to categorical or continuous
        values, or according to a pre-defined sequence of colors. Additionally,
        edges can optionally be superimposed.

        The ``embed`` method must be called sometime before calling this method,
        with a non-None value for the ``snapshot_every`` keyword argument.

        If you want to save the GIF (instead of just viewing it in a Jupyter
        notebook), make sure to supply a path via the ``savepath`` keyword
        argument.

        Arguments
        ---------
        color_by: np.ndarray(shape=mde.n_items), optional
            A sequence of values, one for each item, which should be
            used to color each embedding vector. These values may either
            be categorical or continuous. For example, if ``n_items`` is 4,

            .. code:: python3

                categorical_values = np.ndarray(['dog', 'cat', 'zebra', 'cat'])
                other_categorical_values = np.ndarray([0, 1, 1, 2]
                continuous_values = np.ndarray([0.1, 0.5, 0.31, 0.99]

            are all acceptable values for this argument. A finite number of
            colors is used when the values are categorical, while a
            spectrum of colors is used when the values are continuous.
        color_map: str or matplotlib colormap instance
            Color map to use when resolving ``color_by`` to colors; ignored
            when ``color_by`` is None.
        colors: np.ndarray(shape=(mde.n_items, 4)), optional
            A sequence of colors, one for each item, specifying the exact
            color each item should be colored. Each row must represent
            an RGBA value.

            Only one of ``color_by`` and ``colors`` should be non-None.
        edges: {torch.Tensor/np.ndarray}(shape=(any, 2)), optional
            List of edges to superimpose over the scatter plot.
        axis_limits: tuple(length=2), optional
            tuple (limit_low, limit_high) of axis limits, applied to both
            the x and y axis.
        background_color: str, optional
            color of background
        marker_size: float, optional
            size of each point in the scatter plot
        figsize_inches: tuple(length=2)
            size of figures in inches: (width_inches, height_inches)
        fps : float, optional
            the number of frames per second at which the movie should be
            shown
        tmpdir: str, optional
            path to directory, where individual images comprising the GIF
            will be stored
        savepath: str, optional
            path at which to save the GIF.
        """
        if self.solve_stats is None or not self.solve_stats.snapshots:
            raise ValueError(
                "The play method can only be called after "
                "calling embed with the snapshot_every keyword argument, "
                "as in mde.embed(snapshot_every=5)"
            )

        from pymde import experiment_utils

        try:
            import PIL
        except ImportError:
            raise ImportError("Install PIL to use this method.")

        try:
            from tqdm.auto import tqdm
        except ImportError:
            tqdm = lambda x: x

        if tmpdir is None:
            tmpdir_obj = tempfile.TemporaryDirectory()
            tmpdir = tmpdir_obj.name

        if axis_limits is None:
            lim_low = torch.tensor(float("inf"))
            lim_high = -torch.tensor(float("inf"))
            for shot in self.solve_stats.snapshots:
                low = shot.min()
                high = shot.max()
                if low < lim_low:
                    lim_low = low
                if high > lim_high:
                    lim_high = high
            axis_limits = experiment_utils._square_lim(lim_low, lim_high)

        for i in tqdm(range(len(self.solve_stats.snapshots) + 1)):
            # Repeat the last frame just once, to make the final embedding
            # look sticky.
            #
            # TODO: doing this in a multiprocessing map hangs when run under
            # a jupyter notebook, and disabling interactive mode (plt.ioff())
            # doesn't fix it
            experiment_utils._plot_gif_frame(
                self.solve_stats.snapshots[
                    min(len(self.solve_stats.snapshots) - 1, i)
                ],
                color_by=color_by,
                cmap=color_map,
                colors=colors,
                s=marker_size,
                lim=axis_limits,
                background_color=background_color,
                title=None,
                edges=edges,
                figsize_inches=figsize_inches,
                i=i,
                outdir=tmpdir,
            )

        if fps is None:
            # aim for a 4 second movie, but no fewer than 15 fps
            n_frames = len(self.solve_stats.snapshots)
            fps = max(n_frames / 4.0, 15.0)

        frame_duration = 1000.0 / fps
        fp_in = tmpdir + "/*.png"
        if savepath is None:
            fp_out = tmpdir + "/mde.gif"
        else:
            fp_out = savepath

        img, *imgs = [PIL.Image.open(f) for f in sorted(glob.glob(fp_in))]
        img.save(
            fp=fp_out,
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=frame_duration,
            loop=0,
        )

        try:
            from IPython.display import Image, display

            with open(fp_out, "rb") as f:
                display(Image(f.read()))
        except ImportError:
            pass
