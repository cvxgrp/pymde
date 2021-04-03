__version__ = "0.1.6"

from pymde.problem import MDE

from pymde.constraints import Centered, Anchored, Standardized

from pymde import datasets

from pymde.preprocess.graph import Graph

from pymde.util import all_edges
from pymde.util import align
from pymde.util import center
from pymde.util import rotate

from pymde.experiment_utils import latexify, plot

from pymde.functions import losses, penalties

from pymde.recipes import preserve_distances, preserve_neighbors
