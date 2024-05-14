# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
import os
import mock
from unittest.mock import MagicMock


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


MOCK_MODULES = [
    "pygtk",
    "h5py",
    "gtk",
    "ase",
    "cv2",
    "gobject",
    "argparse",
    "numpy",
    "scipy",
    "pandas",
    "skimage",
    "pyfftw",
    "pyfftw.interfaces",
    "scikit-image",
    "matplotlib.axes",
    "numba",
    "imagecodecs",
    "scipy.optimize",
    "scipy.ndimage",
    "matplotlib.pyplot",
    "matplotlib.colors",
    "scipy.interpolate",
    "skimage.feature",
    "pywt",
    "pywavelets",
    "matplotlib.image",
    "matplotlib",
    "matplotlib.colors",
    "matplotlib.gridspec",
    "matplotlib.cm",
    "skimage.color",
    "scipy.signal",
    "scipy.misc",
    "mpl_toolkits.axes",
    "mpl_toolkits.axes_grid1",
    "scipy.interpolate",
    "skimage.restoration",
    "scipy.special",
    "matplotlib.gridspec",
    "matplotlib_scalebar",
    "matplotlib_scalebar.scalebar",
    "numpy.core.multiarray",
    "matplotlib.offsetbox",
    "multiprocessing",
    "dask.array",
    "dask.distributed",
    "dask",
    "cv2",
    "pyfftw.interfaces.numpy_fft",
    "numexpr",
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

import matplotlib

matplotlib.use("agg")
import sphinx
import sphinx_rtd_theme

sys.path.append(os.path.abspath("../"))
import stemtool


# -- Project information -----------------------------------------------------

project = "STEMTool"
copyright = "2021, Debangshu Mukherjee"
author = "Debangshu Mukherjee"
master_doc = "index"

# The full version, including alpha/beta/rc tags
release = stemtool.__version__
autoapi_dirs = ["../stemtool"]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "recommonmark",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# General information about the project.


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
add_function_parentheses = True
add_module_names = True
show_authors = True
source_suffix = ".rst"


autoclass_content = "both"

autodoc_default_flags = [
    "members",
    "inherited-members",
    # 'private-members',
    # 'show-inheritance'
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
