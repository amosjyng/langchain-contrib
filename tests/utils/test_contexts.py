"""Test context managers."""
import os

import pytest

from langchain_contrib.utils import temporary_file


def test_temporary_file_asserts() -> None:
    """Test that temporary_file will raise if file is not created."""
    with pytest.raises(AssertionError):
        with temporary_file("non-existent.txt"):
            pass


def test_temporary_file_no_asserts() -> None:
    """Test that temporary_file will not raise if ordered not to."""
    with temporary_file("non-existent.txt", check_creation=False):
        pass


def test_temporary_file_removed() -> None:
    """Test that temporary_file will remove the created file."""
    temporary_file_path = "created.txt"
    assert not os.path.isfile(temporary_file_path)

    with temporary_file(temporary_file_path):
        with open(temporary_file_path, "w") as f:
            f.write("Testing")
        assert os.path.isfile(temporary_file_path)

    assert not os.path.isfile(temporary_file_path)
