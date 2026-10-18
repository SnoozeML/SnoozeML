# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import os
from pathlib import Path
import sys

# Add python/ directory to sys.path so autodoc can import snoozeml
SDK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SDK_ROOT))

project = "SnoozeML Python SDK"
copyright = "2026, SnoozeML Contributors"
author = "SnoozeML Contributors"
version = "0.1.0"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# HTML output settings with fallback if shibuya is not installed
try:
    import shibuya  # noqa: F401

    html_theme = "shibuya"
    html_theme_options = {
        "nav_links": [
            {
                "title": "← Back to SnoozeML Main Docs",
                "url": "../",
                "resource": True,
            },
            {
                "title": "Python SDK Guide",
                "url": "../client/python-sdk/",
                "resource": True,
            },
        ],
    }
except ImportError:
    html_theme = "alabaster"
    html_theme_options = {}

html_static_path = ["_static"]
html_title = "SnoozeML Python SDK Documentation"

# Cross-link back to the root SnoozeML Core Service documentation
html_context = {
    "display_github": True,
    "github_user": "SnoozeML",
    "github_repo": "SnoozeML",
    "github_version": "main",
    "conf_py_path": "/python/docs/",
    "core_docs_url": "../",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
