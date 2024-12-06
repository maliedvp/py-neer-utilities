import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from src import neer_match_utilities

# -- Project information -----------------------------------------------------
project = "Neer Match Utilities"
copyright = "2024, Pantelis Karapanagiotis, Marius Liebald"
author = "Pantelis Karapanagiotis, Marius Liebald"
release = neer_match_utilities.__version__

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []

myst_enable_extensions = ["dollarmath", "colon_fence"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

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
    app.connect("autodoc-skip-member", _skip)

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/extra.css",
    "https://cdn.datatables.net/2.1.8/css/dataTables.dataTables.min.css",
]

html_js_files = [
    "https://ajax.googleapis.com/ajax/libs/jquery/1.11.2/jquery.min.js",
    "https://cdn.datatables.net/2.1.8/js/dataTables.min.js",
]
html_logo = "_static/img/logo.png"
html_favicon = "_static/img/favicon/favicon.ico"

html_theme_options = {
    "collapse_navigation": False,  # Expand navigation by default
    "sticky_navigation": True,    # Keep navigation fixed during scroll
    "navigation_depth": 4,        # Number of levels shown in the sidebar
    "titles_only": False,         # Only display the page title in the sidebar
}
