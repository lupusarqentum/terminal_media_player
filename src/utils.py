import sys


def print_error(message: str) -> None:
    """Prints red-colored error message to stderr.

    Parameters:
        message: A message to be printed.

    Returns:
        None.
    """
    print("\033[31m" + message + "\033[39m", file=sys.stderr)


