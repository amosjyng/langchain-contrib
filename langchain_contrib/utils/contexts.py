"""Module to change the current directory."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Optional


@contextmanager
def current_directory(path: Optional[str] = None) -> Generator:
    """Change the current directory for the duration of this context.

    Restores the current directory back to its original value upon exiting the context,
    even if more directory changes have been made in the meantime. If no path is
    provided as input, the current directory will be saved and restored on context
    exit.
    """
    if path is None:
        path = "."

    og_cwd: str = os.getcwd()
    os.makedirs(path, exist_ok=True)
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(og_cwd)


@contextmanager
def temporary_file(path: str, check_creation: bool = True) -> Generator:
    """Ensure that a temporarily created file will stop existing on exit.

    Useful for cleanup after tests. By default, this also checks that the file will
    have been successfully created during the test.
    """
    if os.path.isfile(path):
        os.remove(path)

    try:
        yield
        assert not check_creation or os.path.isfile(path)
    finally:
        if os.path.isfile(path):
            os.remove(path)
