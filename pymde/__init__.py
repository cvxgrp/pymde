__version__ = "0.2.3"


def _fix_omp():
    """Fix OpenMP duplicate-library conflict between torch and faiss on macOS.

    Both packages bundle their own libomp.dylib. Loading both in the same
    process aborts with OMP Error #15. We resolve this by making faiss's
    copy a symlink to torch's, so only one copy is loaded at runtime.
    """
    import sys

    if sys.platform != "darwin":
        return

    import os
    import pathlib

    try:
        import torch
        import importlib.util
        faiss_spec = importlib.util.find_spec("faiss")
    except ImportError:
        return

    if faiss_spec is None or faiss_spec.submodule_search_locations is None:
        return

    torch_omp = (
        pathlib.Path(torch.__file__).parent / "lib" / "libomp.dylib"
    )
    faiss_dir = pathlib.Path(faiss_spec.submodule_search_locations[0])
    faiss_omp = faiss_dir / ".dylibs" / "libomp.dylib"

    if not torch_omp.exists() or not faiss_omp.exists():
        return
    if faiss_omp.is_symlink() or os.path.samefile(torch_omp, faiss_omp):
        return

    try:
        faiss_omp.unlink()
        faiss_omp.symlink_to(torch_omp)
    except OSError:
        pass


_fix_omp()


from pymde.problem import MDE

from pymde.constraints import Centered, Anchored, Standardized

from pymde import datasets

from pymde.preprocess.graph import Graph

from pymde.util import all_edges
from pymde.util import align
from pymde.util import center
from pymde.util import rotate
from pymde.util import seed

from pymde.experiment_utils import latexify, plot

from pymde.functions import losses, penalties

from pymde.quadratic import pca
from pymde.recipes import (
    laplacian_embedding,
    preserve_distances,
    preserve_neighbors,
)
