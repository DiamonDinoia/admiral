project = "Admiral"
copyright = "2026, Marco Barbone"
author = "Marco Barbone"
release = "0.0.1"

extensions = [
    "myst_parser",
    "sphinx_rtd_theme",
]

exclude_patterns = ["_build"]

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

myst_heading_anchors = 3
