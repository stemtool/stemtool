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

MOCK_MODULES = ['pygtk', 
                'gtk', 
                'gobject', 
                'argparse', 
                'numpy',
                'scipy',
                'pandas', 
                'skimage', 
                'pyfftw',
                'pyfftw.interfaces',
                'scikit-image',
                'matplotlib.axes',
                'numba', 
                'imagecodecs',
                'matplotlib', 
                'matplotlib.pyplot',
                'matplotlib.colors',
                'scipy.interpolate', 
                'skimage.feature', 
                'pywt', 
                'pywavelets']

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

from unittest.mock import MagicMock
import matplotlib
matplotlib.use('agg')
import sphinx
import sphinx_rtd_theme
sys.path.append(os.path.abspath('../'))
import stemtool


# -- Project information -----------------------------------------------------

project = u'stemtool'
copyright = u'2020, Debangshu Mukherjee'
author = u'Debangshu Mukherjee'

# The full version, including alpha/beta/rc tags
release = stemtool.__version__
autoapi_dirs = ['../stemtool']

#Add Mock files to fool RTD for C dependent packages

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
            return MagicMock()

MOCK_MODULES = ['pygtk', 
                'gtk', 
                'gobject', 
                'argparse', 
                'numpy', 
                'pandas', 
                'skimage', 
                'pyfftw', 
                'scikit-image', 
                'numba', 
                'imagecodecs']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'recommonmark',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# General information about the project.


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
add_function_parentheses = True
add_module_names = True
show_authors = True
source_suffix = '.rst'


autoclass_content = 'both'

autodoc_default_flags = ['members',
                         'inherited-members',
                         # 'private-members',
                         # 'show-inheritance'
                         ]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
