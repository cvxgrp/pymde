"""Datasets

Module for downloading datasets.

The functions in this module return a Dataset instance. Each function
takes at least two arguments:

    root:
        path to directory in which data should be downloaded (default './')
    download:
        bool indicating whether to download data if not present
        (default True)
"""

import os

import numpy as np
from pymde.preprocess.graph import Graph
from pymde.problem import LOGGER
import scipy.sparse as sp
import torch
import torchvision


def _is_cached(root, files):
    return all(os.path.exists(os.path.join(root, fname)) for fname in files)


def _install_headers():
    # install simple header to bypass mod_security
    import urllib

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)


class Dataset(object):
    """Represents a dataset.

    Each instance has two main attrs:

        data:
            either a data matrix, or an instance of pymde.Graph; the
            data to be embedded.

        attributes:
            a dictionary whose values give attributes associated
            with the items in the data, such as the digit labels in MNIST.

    Other data that the dataset might carry is in the (dict) attribute
    `other_data`. Metadata about where the data came from is stored in
    `metadata`.
    """

    def __init__(self, data, attributes, other_data=None, metadata=None):
        self.data = data
        self.attributes = attributes
        self.other_data = other_data if other_data is not None else {}
        self.metadata = metadata if metadata is not None else {}

    @property
    def attribute(self):
        if len(self.attributes) == 1:
            return list(self.attributes.values())[0]


def FashionMNIST(root="./", download=True) -> Dataset:
    """Fashion-MNIST dataset (Xiao, et al.).

    The Fashion-MNIST dataset contains 70,000, 28x28 images of
    Zalando fashion articles.

    - ``data``: ``torch.Tensor`` with 70,000 rows, each of
      length 784 (representing the pixels in the image).
    - ``attributes`` dict: the key ``class`` holds an array
      in which each entry gives the fashion article in the corresponding row of
      ``data``.
    """
    root = os.path.expanduser(root)

    extract_root = os.path.join(root, "FashionMNIST")
    raw = os.path.join(extract_root, "raw")
    processed = os.path.join(extract_root, "processed")
    raw_files = [
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    ]
    processed_files = ["test.pt", "training.pt"]

    def load_dataset():
        f_mnist_train = torchvision.datasets.FashionMNIST(
            root=root,
            train=True,
            download=True,
        )
        f_mnist_test = torchvision.datasets.FashionMNIST(
            root=root,
            train=False,
            download=True,
        )

        images = torch.cat([f_mnist_train.data, f_mnist_test.data])
        label = torch.cat([f_mnist_train.targets, f_mnist_test.targets])
        attributes = {"class": label}
        return Dataset(
            data=images.reshape(-1, 784),
            attributes=attributes,
            metadata={
                "authors": "Xiao, et al.",
                "url": "https://github.com/zalandoresearch/fashion-mnist",
            },
        )

    if _is_cached(raw, raw_files) and _is_cached(processed, processed_files):
        LOGGER.info("Loading cached dataset.")
        return load_dataset()

    if not download:
        raise RuntimeError("`download` is False, but data is not cached.")

    _install_headers()

    return load_dataset()


def MNIST(root="./", download=True) -> Dataset:
    """MNIST dataset (LeCun, et al.).

    The MNIST dataset contains 70,000, 28x28 images of handwritten digits.

    - ``data``: ``torch.Tensor`` with 70,000 rows, each of
      length 784 (representing the pixels in the image).
    - ``attributes`` dict: the key ``digits`` holds an array
      in which each entry gives the digit depicted in the corresponding row of
      ``data``.
    """
    url = "https://akshayagrawal.com/mnist/MNIST.tar.gz"

    root = os.path.expanduser(root)

    extract_root = os.path.join(root, "MNIST")
    raw = os.path.join(extract_root, "raw")
    processed = os.path.join(extract_root, "processed")
    raw_files = [
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    ]
    processed_files = ["test.pt", "training.pt"]

    def load_dataset():
        mnist_train = torchvision.datasets.MNIST(
            root=root,
            train=True,
            download=False,
        )
        mnist_test = torchvision.datasets.MNIST(
            root=root,
            train=False,
            download=False,
        )

        images = torch.cat([mnist_train.data, mnist_test.data])
        digits = torch.cat([mnist_train.targets, mnist_test.targets])
        attributes = {"digits": digits}
        return Dataset(
            data=images.reshape(-1, 784),
            attributes=attributes,
            metadata={
                "authors": "LeCunn, et al.",
                "url": "http://yann.lecun.com/exdb/mnist/",
            },
        )

    if _is_cached(raw, raw_files) and _is_cached(processed, processed_files):
        LOGGER.info("Loading cached dataset.")
        return load_dataset()

    if not download:
        raise RuntimeError("`download` is False, but data is not cached.")

    _install_headers()
    filename = url.rpartition("/")[2]
    LOGGER.info("Downloading MNIST dataset ...")
    torchvision.datasets.utils.download_and_extract_archive(
        url, download_root=root, filename=filename
    )
    os.remove(os.path.join(root, filename))
    LOGGER.info("Download complete.")

    return load_dataset()


def google_scholar(root="./", download=True, full=False) -> Dataset:
    """Google Scholar dataset (Agrawal, et al.).

    The Google Scholar dataset contains an academic coauthorship graph: the
    nodes are authors, and two authors are connected by an edge if either
    author listed the other as a coauthor on Google Scholar. (Note that if
    two authors collaborated on a paper, but neither has listed the other
    as a coauthor on their Scholar profiles, then they will not be connected
    by an edge).

    If ``full`` is False, obtains a small version of the dataset, on roughly
    40,000 authors, each with h-index at least 50. If ``full`` is True,
    obtains the whole dataset, on roughly 600,000 authors. The full dataset
    is roughly 1GB in size.

    - ``data``: a ``pymde.Graph``, with nodes representing authors
    - ``attributes``: the ``coauthors`` key has an array holding the number
      of coauthors of each other, normalized to be a percentile.
    - ``other_data``: holds a dataframe describing the dataset, keyed by
      ``dataframe``.
    """

    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Please install pandas.")

    root = os.path.expanduser(root)

    if not full:
        url = "https://akshayagrawal.com/scholar/google_scholar.tar.gz"
    else:
        url = "https://akshayagrawal.com/scholar/google_scholar_full.tar.gz"

    extract_root = os.path.join(root, "scholar/")
    data_file = os.path.join(extract_root, "scholar_graph.npz")
    dataframe_file = os.path.join(extract_root, "scholar_df.pkl")
    if not full:
        disciplines_dataframe_file = os.path.join(
            extract_root, "scholar_disciplines_df.pkl"
        )

    def load_dataset():
        data = Graph(sp.load_npz(os.path.join(root, data_file)))
        df = pd.read_pickle(os.path.join(root, dataframe_file))
        attributes = {"coauthors": df["coauthors_pct"]}
        if not full:
            disciplines_df = pd.read_pickle(
                os.path.join(root, disciplines_dataframe_file)
            )
            other_data = {"dataframe": df, "disciplines": disciplines_df}
        else:
            other_data = None
        metadata = {"authors": "Agrawal"}
        return Dataset(data, attributes, other_data, metadata)

    if not full:
        files = [
            data_file,
            dataframe_file,
            disciplines_dataframe_file,
        ]
    else:
        files = [
            data_file,
            dataframe_file,
        ]

    if _is_cached(root, files):
        LOGGER.info("Loading cached dataset.")
        return load_dataset()

    if not download:
        raise RuntimeError("`download` is False, but data is not cached.")

    _install_headers()

    filename = url.rpartition("/")[2]
    torchvision.datasets.utils.download_and_extract_archive(
        url, download_root=root, extract_root=extract_root, filename=filename
    )
    os.remove(os.path.join(root, filename))
    LOGGER.info("Dataset is now available.")
    return load_dataset()


def covid19_scrna_wilk(root="./", download=True) -> Dataset:
    """COVID-19 scRNA data (Wilk, et al.).

    The COVID-19 dataset includes a PCA embedding of single-cell
    mRNA transcriptomes of roughly 40,000 cells, taken from some patients
    with COVID-19 infections and from healthy controls.

    Instructions on how to obtain the full dataset are available in the
    Wilk et al. paper: https://www.nature.com/articles/s41591-020-0944-y,

    - ``data``: the PCA embedding
    - ``attributes``: two keys, ``cell_type`` and ``health_status``.
    """
    root = os.path.expanduser(root)

    url = "https://akshayagrawal.com/scrna/scrna_covid19_wilk.tar.gz"

    extract_root = os.path.join(root, "scrna_covid19_wilk/")
    data_file = os.path.join(extract_root, "scrna_data_matrix.npy")
    cell_type_attr_file = os.path.join(extract_root, "scrna_cell_type_attr.npy")
    health_status_attr_file = os.path.join(
        extract_root, "scrna_health_status_attr.npy"
    )

    def load_dataset():
        data = np.load(os.path.join(root, data_file))
        attributes = {
            "cell_type": np.load(
                os.path.join(root, cell_type_attr_file), allow_pickle=True
            ),
            "health_status": np.load(
                os.path.join(root, health_status_attr_file), allow_pickle=True
            ),
        }
        metadata = {
            "authors": "Wilk, et al.",
            "url": "https://www.nature.com/articles/s41591-020-0944-y",
        }
        return Dataset(data, attributes, metadata=metadata)

    files = [
        data_file,
        cell_type_attr_file,
        health_status_attr_file,
    ]

    if _is_cached(root, files):
        LOGGER.info("Loading cached dataset.")
        return load_dataset()

    if not download:
        raise RuntimeError("`download` is False, but data is not cached.")

    _install_headers()

    filename = url.rpartition("/")[2]
    torchvision.datasets.utils.download_and_extract_archive(
        url, download_root=root, extract_root=extract_root, filename=filename
    )
    os.remove(os.path.join(root, filename))
    LOGGER.info("Dataset is now available.")
    return load_dataset()


def population_genetics(root="./", download=True) -> Dataset:
    """Population genetics dataset (Nelson, et al)

    The population genetics dataset includes a PCA embedding (in R^20) of
    single nucleotide polymorphism data associated with 1,387
    individuals thought to be of European descent. (The data is from the
    Population Reference Sample project by Nelson, et al.)

    It also contains a "corrupted" version of the data, in which 154 additional
    points have been injected; the first 10 coordinates of these synthetic
    points are generated using a discrete uniform distribution on
    {0, 1, 2}, and the last 10 are generated using a discrete uniform
    distritubtion on {1/12, /18}.

    A study of Novembre et al (2008) showed that a PCA embedding in R^2
    roughly resembles the map of Europe, suggesting that the genes
    encode geographical information. But PCA does not produce interesting
    visualizations of the corrupted data. If distortion functions are chosen to
    be robust (eg, using the Log1p or Huber attractive penalties), we can
    create embeddings that preserve the geographical structure, while
    placing the synthetic points to the side, in their own cluser.

    - ``data``: the PCA embedding of the clean genetic data, in R^20
    - ``corrupted_data``: the corrupted data, in R^20
    - ``attributes``: two keys, ``clean_colors`` and ``corrupted_colors``.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Please install pandas.")

    root = os.path.expanduser(root)

    url = "https://akshayagrawal.com/popres/popres.tar.gz"

    extract_root = os.path.join(root, "population_genetics/")
    data_file = os.path.join(extract_root, "data.csv")
    corrupted_data_file = os.path.join(extract_root, "corrupted_data.csv")
    clean_attr_file = os.path.join(extract_root, "data_colors.csv")
    corrupted_attr_file = os.path.join(extract_root, "corrupted_colors.csv")

    def load_dataset():
        data = pd.read_csv(os.path.join(root, data_file)).to_numpy()
        corrupted_data = pd.read_csv(
            os.path.join(root, corrupted_data_file)
        ).to_numpy()
        attributes = {
            "clean_colors": pd.read_csv(
                os.path.join(root, clean_attr_file)
            ).to_numpy(),
            "corrupted_colors": pd.read_csv(
                os.path.join(root, corrupted_attr_file)
            ).to_numpy(),
        }
        metadata = {
            "authors": "Nelson, et al",
            "url": "https://pubmed.ncbi.nlm.nih.gov/18760391/",
        }
        dataset = Dataset(data, attributes, metadata=metadata)
        dataset.corrupted_data = corrupted_data
        return dataset

    files = [
        data_file,
        corrupted_data_file,
        clean_attr_file,
        corrupted_attr_file,
    ]

    if _is_cached(root, files):
        LOGGER.info("Loading cached dataset.")
        return load_dataset()

    if not download:
        raise RuntimeError("`download` is False, but data is not cached.")

    _install_headers()

    filename = url.rpartition("/")[2]
    torchvision.datasets.utils.download_and_extract_archive(
        url, download_root=root, extract_root=extract_root, filename=filename
    )
    os.remove(os.path.join(root, filename))
    LOGGER.info("Dataset is now available.")
    return load_dataset()


def counties(root="./", download=True) -> Dataset:
    """US counties (2013-2017 ACS 5-Year Estimates)

    This dataset contains 34 demographic features for each of the 3,220 US
    counties. The features were collected by the 2013-2017 ACS 5-Year
    Estimates longitudinal survey, run by the US Census Bureau.

    - ``data``: the PCA embedding of the clean genetic data, in R^20
    - ``county_dataframe``: the raw ACS data
    - ``voting_dataframe``: the raw 2016 voting data
    - ``attributes``: one key, ``democratic_fraction``, the fraction of
      of voters who voted Democratic in each county in the 2016 presidential
      election
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Please install pandas.")

    root = os.path.expanduser(root)

    url = "https://akshayagrawal.com/voting/voting.tar.gz"

    extract_root = os.path.join(root, "counties/")
    data_file = os.path.join(extract_root, "county_data.npy")
    voting_attr_file = os.path.join(extract_root, "democratic_fraction.npy")

    county_df_file = os.path.join(extract_root, "acs2017_county_data.csv")
    voting_df_file = os.path.join(
        extract_root, "2016_US_County_Level_Presidential_Results.csv"
    )

    def load_dataset():
        data = np.load(os.path.join(root, data_file), allow_pickle=True)
        county_dataframe = pd.read_csv(os.path.join(root, county_df_file))
        voting_dataframe = pd.read_csv(os.path.join(root, voting_df_file))
        attributes = {
            "democratic_fraction": np.load(os.path.join(root, voting_attr_file))
        }
        metadata = {
            "authors": "ACS 2013-2017 survey",
            "url": "https://www.census.gov/newsroom/press-kits/2018/acs-5year.html",  # noqa: E501
        }
        dataset = Dataset(data, attributes, metadata=metadata)
        dataset.county_dataframe = county_dataframe
        dataset.voting_dataframe = voting_dataframe
        return dataset

    files = [
        data_file,
        voting_attr_file,
        county_df_file,
        voting_attr_file,
    ]

    if _is_cached(root, files):
        LOGGER.info("Loading cached dataset.")
        return load_dataset()

    if not download:
        raise RuntimeError("`download` is False, but data is not cached.")

    _install_headers()

    filename = url.rpartition("/")[2]
    torchvision.datasets.utils.download_and_extract_archive(
        url, download_root=root, extract_root=extract_root, filename=filename
    )
    os.remove(os.path.join(root, filename))
    LOGGER.info("Dataset is now available.")
    return load_dataset()
