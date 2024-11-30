import os
import sphinx_bootstrap_theme
import sys


sys.path.insert(0, os.path.abspath("../.."))

from src import neer_match_utilities


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Neer Match Utilities'
copyright = '2024, Pantelis Karapanagiotis, Marius Liebald'
author = 'Pantelis Karapanagiotis, Marius Liebald'
release = neer_match_utilities.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	"myst_parser",
    'sphinx.ext.autodoc',   # Extract docstrings from your code
    'sphinx.ext.napoleon',  # Parse Google-style or NumPy-style docstrings
    'sphinx.ext.viewcode',  # Add links to highlighted source code
    "sphinx.ext.extlinks",
    "sphinx_autodoc_typehints",
]

templates_path = ['_templates']
exclude_patterns = []

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


extlinks = {
    "ltn-tut-2c": (
        "https://nbviewer.org/github/logictensornetworks/logictensornetworks/blob/"
        "master/tutorials/2b-operators_and_gradients.ipynb%s",
        None,
    )
}


def _skip(app, what, name, obj, would_skip, options):
    if name in ["__getitem__", "__init__", "__iter__", "__len__", "__str__"]:
        return False
    return would_skip


def setup(app):
    """Set up the Sphinx application."""
    app.connect("autodoc-skip-member", _skip)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "bootstrap"
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

html_static_path = ["_static"]

html_logo = "_static/img/logo.png"
html_theme_options = {
    "github_user": "pi-kappa-devel",
    "extra_nav_links": {
        "Relevant Links": "#",
        "📦 PyPi Package": "https://pypi.org/project/neer-match/",
        "📦 R Package": "https://github.com/pi-kappa-devel/r-neer-match",
        "📖 R Docs": "https://r-neer-match.pikappa.eu",
    },
}

html_favicon = "_static/img/favicon/favicon.ico"

html_css_files = [
    "css/extra.css",
    "https://cdn.datatables.net/2.1.8/css/dataTables.dataTables.min.css",
]

html_js_files = [
    "https://ajax.googleapis.com/ajax/libs/jquery/1.11.2/jquery.min.js",
    "https://cdn.datatables.net/2.1.8/js/dataTables.min.js",
]
