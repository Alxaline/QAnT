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
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
print(os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "QAnT"
copyright = "2021, Alexandre CARRE"
author = "Alexandre CARRE"

# The full version, including alpha/beta/rc tags
release = "0.0.1"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "recommonmark",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
]

# The master toctree document.
master_doc = "index"

# The suffix of source filenames.
source_suffix = ".rst"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    "collapse_navigation": True,
    "display_version": True,
    "sticky_navigation": True,  # Set to False to disable the sticky nav while scrolling.
    "logo_only": True,  # if we have a html_logo below, this shows /only/ the logo with no title text
    "style_nav_header_background": "#FBFBFB",
}
html_context = {
    "display_github": True,
    "github_user": "Alxaline",
    "github_repo": "QAnT",
    "github_version": "master",
    "conf_py_path": "/docs/",
}
html_scaled_image_link = False
html_show_sourcelink = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# -- Options for LaTeX output ------------------------------------------------
latex_engine = "pdflatex"
latex_elements = {
    # The paper size ("letterpaper" or "a4paper").
    #
    "papersize": "a4paper",

    # The font size ("10pt", "11pt" or "12pt").
    #
    "pointsize": "10pt",

    # Additional stuff for the LaTeX preamble.
    #
    "preamble": "",

    # Latex figure (float) alignment
    #
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "QAnT.tex", "QAnT Documentation",
     "Alexandre CARRE", "manual"),
]
