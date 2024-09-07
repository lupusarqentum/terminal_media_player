import sys


def print_error(message):
    print("\033[31m" + message + "\033[39m", file=sys.stderr)


