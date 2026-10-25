project = "Admiral"
copyright = "2026, Marco Barbone"
author = "Marco Barbone"

extensions = [
    "myst_parser",
    "sphinx_rtd_theme",
]

exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
}

html_context = {
    "display_github": True,
    "github_user": "DiamonDinoia",
    "github_repo": "admiral",
    "github_version": "master",
    "conf_py_path": "/docs/",
}
