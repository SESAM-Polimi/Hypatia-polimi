"""Sphinx configuration for the Hypatia documentation.

Build locally from the repository root:

    pip install -r doc/requirements.txt
    sphinx-build -W -b html doc/source doc/build/html

-W turns warnings into errors, as on Read the Docs, so broken references and
missing pages fail the build instead of producing a silently incomplete site.
"""
import os
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "doc", "tools"))

# Parameter tables are generated from Hypatia's own templates at every build
# (doc/tools/reference.py), so they cannot drift from the code.
import reference  # noqa: E402

reference.write(os.path.join(HERE, "_generated"))

project = "Hypatia"
author = "SESAM group, Department of Energy, Politecnico di Milano"
copyright = f"2021-{date.today().year}, SESAM, Politecnico di Milano"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_parser",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["dollarmath", "colon_fence"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "_generated"]
root_doc = "index"

# References to numbered equations with :eq:`label`.
math_eqref_format = "({number})"
numfig = True

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosummary_generate = True
napoleon_numpy_docstring = True

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["css/wrapper.css"]
html_theme_options = {"navigation_depth": 3, "collapse_navigation": False}
html_title = "Hypatia documentation"

# LaTeX/PDF output (sphinx-build -b latex, then pdflatex).
latex_elements = {
    # Long sheet names (Max_production_share_Conv_plus) may break after an
    # underscore, so that they fit in the narrow columns of the input tables.
    "preamble": r"""\usepackage{amsmath,amssymb}
\let\hypatiaunderscore\_
\renewcommand{\_}{\hypatiaunderscore\allowbreak}
""",
    # LaTeX's built-in fonts, so the PDF builds with a minimal TeX installation.
    "fontpkg": "",
}
latex_documents = [
    (root_doc, "hypatia.tex", "Hypatia documentation", author, "manual"),
    # The model formulation on its own, e.g. as an appendix to a paper.
    ("formulation/index", "hypatia-formulation.tex", "Hypatia: model formulation", author, "howto"),
]
