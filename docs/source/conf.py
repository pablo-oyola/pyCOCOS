"""
Sphinx configuration for pyCOCOS documentation.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


project = "pyCOCOS"
author = "superNOVA contributors"
copyright = "2026, superNOVA contributors"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = []
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# The README and CONVENTIONS.md are included as Markdown, not copied.
myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 3

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = True
# Render "Attributes" sections as fields: several of them name properties
# that autodoc also documents, which would describe the same object twice.
napoleon_use_ivar = True

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
}
html_static_path = ["_static"]
