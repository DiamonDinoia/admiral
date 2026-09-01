import os

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

exclude_patterns = ["README.md", "_build"]

_docs_dir = os.path.dirname(os.path.abspath(__file__))
breathe_projects = {"Admiral": os.path.join(_docs_dir, "_doxygen", "xml")}
breathe_default_project = "Admiral"

exhale_args = {
    "containmentFolder": "./api",
    "rootFileName": "library_root.rst",
    "doxygenStripFromPath": os.path.dirname(_docs_dir),
    "rootFileTitle": "API Reference",
    "createTreeView": True,
    "exhaleExecutesDoxygen": True,
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

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
}

html_context = {
    "display_github": True,
    "github_user": "DiamonDinoia",
    "github_repo": "admiral",
    "github_version": "master",
    "conf_py_path": "/docs/",
}

suppress_warnings = [
    "docutils",
    "cpp.duplicate_declaration",
    "toc.not_included",
]

myst_heading_anchors = 3
