# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# RTD environment detection (optional, for any future customizations)
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
rtd_version = os.environ.get("READTHEDOCS_VERSION", "latest")

# -- Project information -----------------------------------------------------

project = "ConfOpt"
copyright = "2025, Riccardo Doyle"
author = "Riccardo Doyle"
release = "2.0.0"
version = "2.0.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "myst_parser",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "tasklist",
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
autodoc_class_attributes = False

# Autosummary settings
autosummary_generate = True
autosummary_generate_overwrite = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "canonical_url": "https://confopt.readthedocs.io/",
    "logo_only": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#2563eb",
    # Toc options - optimized for better navigation
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 3,
    "includehidden": True,
    "titles_only": False,
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# GitHub integration
html_context = {
    "display_github": True,
    "github_user": "rick12000",
    "github_repo": "confopt",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Custom logo and favicon
html_logo = "../assets/logo.png"
html_favicon = None  # RTD will handle this

# The root toctree document (updated from deprecated master_doc)
root_doc = "index"

# The name of the Pygments (syntax highlighting) style to use
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing
todo_include_todos = False

# Security and performance improvements
tls_verify = True
tls_cacerts = ""

# Suppress warnings for external references that may not always be available
suppress_warnings = [
    "ref.doc",
    "ref.ref",
]

# Enable nitpicky mode for better link validation (but suppress known issues)
nitpicky = True
nitpick_ignore = [
    ("py:class", "type"),
    ("py:class", "object"),
    ("py:class", "callable"),
    ("py:class", "default=100"),
    ("py:class", "default=None"),
    ("py:class", "default=15"),
    ("py:class", "default=1"),
    ("py:class", "default=True"),
    ("py:class", "confopt.wrapping.IntRange"),
    ("py:class", "confopt.wrapping.FloatRange"),
    ("py:class", "confopt.wrapping.CategoricalRange"),
    ("py:class", "BaseConformalSearcher"),
    ("py:class", "numpy.array"),
    ("py:class", "sklearn.preprocessing._data.StandardScaler"),
    ("py:class", "confopt.selection.acquisition.BaseConformalSearcher"),
    ("py:class", "confopt.utils.tracking.ProgressBarManager"),
]

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
}

latex_documents = [
    (root_doc, "confopt.tex", "ConfOpt Documentation", author, "manual"),
]

# -- Options for manual page output ------------------------------------------

man_pages = [(root_doc, "confopt", "ConfOpt Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        root_doc,
        "confopt",
        "ConfOpt Documentation",
        author,
        "confopt",
        "Voice Command Assistant for Accessibility.",
        "Miscellaneous",
    ),
]

# -- Options for Epub output -------------------------------------------------

epub_title = project
epub_exclude_files = ["search.html"]
