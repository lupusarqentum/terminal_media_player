# TerminalVideoPlayer, a program using command line interface to play videos.
# Copyright (C) 2024  Roman Lisov
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# https://www.gnu.org/licenses/gpl-3.0.html

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
