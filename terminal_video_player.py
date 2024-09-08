#!/usr/bin/env python3

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

import sys

from src.configuration import Configuration
from src.image_processing import ImageProcessor
from src.utils import print_error, print_warn, get_terminal_size

if __name__ == "__main__":
    CONFIG_LOCATION_PREFIX = "./"
    config = Configuration()
    config_location = CONFIG_LOCATION_PREFIX + "config.json"
    if not config.read_and_apply_JSON_config(config_location):
        print_error("An error occurred when trying to apply config. \
Fallback to default config instead")
        if not config.read_and_apply_JSON_config(config_location):
            print_error("An error occurred when trying to apply \
default config. Can't operate")
            sys.exit(-1)

    if len(sys.argv) < 2:
        print_warn("No input file path was provided. \
Assuming examples/input.png")
        target_file_path = "examples/input.png"
    else:
        target_file_path = sys.argv[1]
        print("Target media path: \"" + target_file_path + "\"")

    terminal_rows, terminal_columns = get_terminal_size()

    image_processor = ImageProcessor(config)
    image_processor.load(target_file_path)
    image_processor.render(terminal_rows, terminal_columns)

    if config.get_boldify():
        print("\033[1m")
        image_processor.display()
        print("\033[0m")
    else:
        image_processor.display()
