import os
import sys

# Make the package importable from the repo root
sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project   = "a6cw"
author    = "Alex"
copyright = "2026, Alex"
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
# galsim and numba require compiled extensions unavailable on the RTD builder;
# mock them so autodoc can import a6cw without errors.
autodoc_mock_imports = ["galsim", "numba"]

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
