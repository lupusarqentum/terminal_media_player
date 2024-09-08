import os
import sys


def print_error(message: str) -> None:
    """Prints red-colored error message to stderr."""
    print("\033[31m" + message + "\033[39m", file=sys.stderr)

def print_warn(message: str) -> None:
    """Prints yellow-colored error message to stdout."""
    print("\033[33m" + message + "\033[39m")

def get_terminal_size() -> tuple:
    """Finds terminal size and returns it as a tuple (rows, columns)."""
    rows, columns = os.popen("stty size", "r").read().split()
    return int(rows), int(columns)

