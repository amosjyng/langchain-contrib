"""Module to remove ANSI escape sequences from strings."""

import re
from typing import List


def ansi_escape_regex() -> re.Pattern:
    """Return a regex pattern to match ANSI escape sequences."""
    # not really ANSI escape sequence, but similar vibe
    rs = r"\r|\\r"

    # how we know we're in an ANSI escape sequence
    escape_prefixes = [r"\\033", r"\\e", r"\x1b"]
    brackets = [r"\[", r"\("]
    all_ansi_escapes = [
        prefix + bracket for prefix in escape_prefixes for bracket in brackets
    ]
    all_escapes_regex = "|".join(all_ansi_escapes)

    # the actual command once we know there's an escape
    escape_command = r"[\d;]*m|\d*[A-K]"

    escapes = f"({all_escapes_regex})({escape_command})"
    return re.compile("|".join([rs, escapes]))


def remove_match(line: str, next_match: re.Match) -> str:
    """Remove the next match from the line."""
    return line[: next_match.start()] + line[next_match.end() :]


def remove_ansi_escapes(input: str) -> str:
    r"""Remove ANSI escape sequences from the input string.

    Assumes that all \r\n have already been replaced by \n.

    Incomplete escape interpreter based on
    https://www.lihaoyi.com/post/BuildyourownCommandLinewithANSIescapecodes.html

    Additional documentation available at
    https://en.wikipedia.org/wiki/ANSI_escape_code#CSI_(Control_Sequence_Introducer)_sequences
    """
    regex = ansi_escape_regex()
    cleaned: List[str] = []
    for line in input.split("\n"):
        next_match = re.search(regex, line)
        while next_match is not None:
            full_match = next_match.group()
            _, command = next_match.groups()
            if full_match == "\r" or full_match == "\\r":
                line = line[next_match.end() :]
            elif command.endswith("A"):
                num_lines = int(command[:-1])
                cleaned = cleaned[:-num_lines]
                line = line[next_match.end() :]
            elif command.endswith("m"):
                line = remove_match(line, next_match)
            elif command.endswith("J"):
                if command == "0J":
                    # todo: track cursor position properly
                    # do nothing because it only affects things after the cursor
                    pass
                line = remove_match(line, next_match)
            elif command.endswith("K"):
                if command == "K" or command == "0K":
                    # erase after cursor. no-op because not tracking cursor
                    line = remove_match(line, next_match)
                else:
                    line = line[next_match.end() :]
            else:  # just remove the ANSI escape code if we can't interpret it
                line = remove_match(line, next_match)
            next_match = re.search(regex, line)
        cleaned.append(line)
    return "\n".join(cleaned)


def interpret_terminal_output(input: str) -> str:
    """Render approximately how terminal output looks on screen."""
    return remove_ansi_escapes(input.replace("\r\n", "\n"))
