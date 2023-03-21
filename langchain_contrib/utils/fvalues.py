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

    joined = ""
    parts = []
    for substring in substrings:
        joined += substring
        if isinstance(substring, F):
            parts.extend(list(substring.parts))
        else:
            parts.append(substring)

        # if it's the empty string, we can just avoid polluting parts with it
        if joiner != "":
            joined += joiner
            parts.append(joiner)

    if joiner != "":  # joiner should only exist in between parts
        joined = joined[: -len(joiner)]
        parts.pop()

    return F(joined, parts=tuple(parts))
