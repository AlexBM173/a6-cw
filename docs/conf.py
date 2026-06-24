import os
import sys

# Anchor to conf.py's own location so this works regardless of where
# sphinx-build is invoked from (local, tox, ReadTheDocs, etc.)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)   # makes `a6cw` and `nfw_theory` importable

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project   = "a6cw"
author    = "Alex Blake Martín"
copyright = "2026, Alex Blake Martín"
release   = "1.0.0"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",        # pull docstrings into API pages
    "sphinx.ext.napoleon",       # Google/NumPy-style docstrings
    "sphinx.ext.mathjax",        # LaTeX math rendering
    "sphinx.ext.viewcode",       # [source] links on API pages
    "sphinx_autodoc_typehints",  # type annotations in signatures
    "myst_parser",               # parse existing Markdown docs
]

# MyST: enable dollar-sign math so the existing docs render correctly
myst_enable_extensions = ["dollarmath", "colon_fence"]
myst_dmath_double_inline = True

# ---------------------------------------------------------------------------
# autodoc settings
# ---------------------------------------------------------------------------
# galsim and numba require compiled extensions unavailable on the RTD builder.
# nfw_theory (repo-root module) is mocked because it imports galsim at the
# top level before Sphinx's mock machinery can intercept it.
autodoc_mock_imports = ["galsim", "numba", "nfw_theory"]

autodoc_default_options = {
    "members":          True,
    "undoc-members":    False,
    "show-inheritance": True,
    "member-order":     "bysource",
}

# Show type hints in the description, not just the signature
autodoc_typehints           = "description"
autodoc_typehints_format    = "short"
always_document_param_types = False

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "titles_only":      False,
}
html_static_path = ["_static"]
html_title       = "a6cw — Weak Lensing Analysis"

# ---------------------------------------------------------------------------
# Source file types
# ---------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md":  "myst",
}
