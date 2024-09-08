import sys


def print_error(message: str) -> None:
    """Prints red-colored error message to stderr."""
    print("\033[31m" + message + "\033[39m", file=sys.stderr)


