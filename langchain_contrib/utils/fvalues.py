"""Module to join F-strings."""

from typing import List, Union

from fvalues import F


def f_join(joiner: str, substrings: List[Union[str, F]]) -> F:
    """Join strings together while preserving their original F and non-F status.

    This function exists to temporarily provide this functionality pending the merge of
    https://github.com/oughtinc/fvalues/pull/11

    Args:
        joiner: The string to join the substrings with.
        substrings: The substrings to join. Can be either regular strings or F-strings.

    Returns:
        The joined string, with all original parts preserved.
    """
    if substrings == []:
        return F("")

    joined = joiner.join(substrings)
    parts = []
    for substring in substrings:
        parts.append(substring)

        # if it's the empty string, we can just avoid polluting parts with it
        if joiner != "":
            parts.append(joiner)

    if joiner != "":  # joiner should only exist in between parts
        parts.pop()

    return F(joined, parts=tuple(parts))
