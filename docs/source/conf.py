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
import furo
import os
import sys
sys.path.insert(0, os.path.abspath('../../../dataprofiler'))
sys.path.insert(0, os.path.abspath('../../../examples'))


# -- Project information -----------------------------------------------------

project = 'Data Profiler'
copyright = '2020, Jeremy Goodsitt, Austin Walters, Anh Truong, Grant Eden, and Chris Wallace'
author = 'Jeremy Goodsitt, Austin Walters, Anh Truong, Grant Eden, and Chris Wallace'

# The full version, including alpha/beta/rc tags
# release = '21.01.20'
from dataprofiler import __version__ as version  # noqa F401


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', 
    'furo', 
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',     
    'nbsphinx',
    'nbsphinx_link',
]

# Don't execute the notebook cells when generating the documentation
# This can be configured on a per notebook basis as well
# See: https://nbsphinx.readthedocs.io/en/0.2.15/never-execute.html#Explicitly-Dis-/
nbsphinx_execute = "never"
nbsphinx_prolog = """
`View this notebook on GitHub <https://github.com/capitalone/rubicon/tree/main/notebooks/{{ env.doc2path(env.docname, base=None) }}>`_
"""

autoclass_content = 'both'
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'inherited-members': True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = f"<div class='hidden'>Data Profiler</div> <div class='version'> v{version[:5]}</div>"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/images/DataProfilerLogoLightTheme.png"
html_theme_options = {
    "light_logo": "images/DataProfilerLogoLightThemeLong.png",
    "dark_logo": "images/DataProfilerDarkLogoLong.png",
}




