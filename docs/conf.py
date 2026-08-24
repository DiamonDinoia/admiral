import os

# Project information
project = "Admiral"
copyright = "2026, Marco Barbone"
author = "Marco Barbone"
release = "0.0.1"

extensions = [
    "breathe",
    "exhale",
    "myst_parser",
    "sphinx_rtd_theme",
]

# The repo-browsing index; the site index is index.rst.
exclude_patterns = ["README.md", "_build"]

# Breathe reads the Doxygen XML that exhale generates below. Anchor to this
# file so the path does not depend on the sphinx-build invocation directory.
_docs_dir = os.path.dirname(os.path.abspath(__file__))
breathe_projects = {"Admiral": os.path.join(_docs_dir, "_doxygen", "xml")}
breathe_default_project = "Admiral"

# Exhale runs Doxygen itself, so `sphinx-build` is the whole docs build:
# no CMake configure and no separate Doxyfile.
exhale_args = {
    "containmentFolder": "./api",
    "rootFileName": "library_root.rst",
    "doxygenStripFromPath": os.path.dirname(_docs_dir),
    "rootFileTitle": "API Reference",
    "createTreeView": True,
    "exhaleExecutesDoxygen": True,
    # The public headers are the top of include/admiral; RECURSIVE = NO keeps
    # include/admiral/detail out. EXTRACT_ALL surfaces overloads that share one
    # comment block with their first declaration.
    "exhaleDoxygenStdin": """
        INPUT            = ../include/admiral
        RECURSIVE        = NO
        FULL_PATH_NAMES  = YES
        EXTRACT_ALL      = YES
        EXTRACT_PRIVATE  = NO
        EXTRACT_STATIC   = NO
        HIDE_FRIEND_COMPOUNDS = YES
        EXCLUDE_SYMBOLS  = detail admiral::detail admiral::detail::* *::detail::*
        ENABLE_PREPROCESSING = YES
        MACRO_EXPANSION  = YES
        EXPAND_ONLY_PREDEF = YES
        PREDEFINED       = ADM_API= \\
                           ADM_C_API= \\
                           ADM_NODISCARD= \\
                           FFTW_C_API= \\
                           FFTW_NODISCARD=
        XML_PROGRAMLISTING = NO
        CASE_SENSE_NAMES = YES
        WARN_IF_UNDOCUMENTED = NO
        QUIET            = YES
    """,
}

# Theme
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
}

# RTD theme reads these to render the "Edit on GitHub" link on every page.
html_context = {
    "display_github": True,
    "github_user": "DiamonDinoia",
    "github_repo": "admiral",
    "github_version": "master",
    "conf_py_path": "/docs/",
}

# Breathe/exhale emit false-positive warnings for overloaded templated APIs.
suppress_warnings = [
    "docutils",
    "cpp.duplicate_declaration",
    "toc.not_included",
]

# The guides link to files outside docs/ (examples, benchmark) with absolute
# GitHub URLs, so myst needs no special handling; heading anchors make the
# usage.md#options style cross-links resolve.
myst_heading_anchors = 3
