"""Module for code that patches various utilities to work with Terminal."""

from . import langchain_patch  # noqa: F401

try:
    from . import vcr_patch  # noqa: F401
except ImportError:
    pass  # vcr_langchain is an optional dependency
