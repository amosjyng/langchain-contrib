"""Configuration file for the Sphinx documentation builder."""

import os
import sys

import toml

project = "langchain-contrib"
copyright = "2023, Amos Ng"
author = "Amos Ng"

# https://stackoverflow.com/a/60159862
sys.path.insert(0, os.path.abspath(".."))

with open("../pyproject.toml") as f:
    data = toml.load(f)

version = data["tool"]["poetry"]["version"]
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"

html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/amosjyng/langchain-contrib",
    "use_repository_button": True,
}

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "amosjyng",  # Username
    "github_repo": "langchain-contrib",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
}

html_static_path = ["_static"]
