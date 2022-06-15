# Configuration file for the Sphinx documentation builder.

import sys
from pathlib import Path
from datetime import datetime
import jupytext
import warnings

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "extensions"))

from HoloNet import __author__, __version__

# -- Project information

project = 'HoloNet'
author = __author__
copyright = f"{datetime.now():%Y}, {author}."
version = __version__

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
api_dir = HERE / "_static" / "api"
api_rel_dir = "_static/api"


nitpicky = True  # Warn about broken links
needs_sphinx = "2.0"  # Nicer param docs

# -- General configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx_autodoc_typehints",
    "scanpydoc",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

autosummary_generate = True
autodoc_member_order = "bysource"
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False

intersphinx_mapping = dict(
    scanpy=("https://scanpy.readthedocs.io/en/stable/", None),
    anndata=("https://anndata.readthedocs.io/en/stable/", None),
    ipython=("https://ipython.readthedocs.io/en/stable/", None),
    matplotlib=("https://matplotlib.org/", None),
    numpy=("https://numpy.org/doc/stable/", None),
    pandas=("https://pandas.pydata.org/pandas-docs/stable/", None),
    python=("https://docs.python.org/3", None),
    scipy=("https://docs.scipy.org/doc/scipy/reference/", None),
    seaborn=("https://seaborn.pydata.org/", None),
    sklearn=("https://scikit-learn.org/stable/", None),
    networkx=("https://networkx.org/documentation/networkx-1.10/", None),
)

typehints_defaults = None

# -- nbsphinx Tutorials ----------------------------------------------------------------

# Enable jupytext notebooks
nbsphinx_custom_formats = {
    ".md": lambda s: jupytext.reads(s, ".md"),
}
# nbsphinx_execute = "always"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]
nbsphinx_timeout = 300


# -- HTML styling ----------------------------------------------------------------------

html_theme = "scanpydoc"
# add custom stylesheet
# https://stackoverflow.com/a/43186995/2340703
html_static_path = ["_static"]
pygments_style = "sphinx"

html_theme_options = dict(navigation_depth=4, logo_only=True)


def setup(app):
    pass


nitpick_ignore = [
    ("py:class", "igraph.Graph"),
    ("py:class", "igraph.Layout"),
    ("py:class", "igraph.layout.Layout"),
]

