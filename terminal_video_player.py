#!/usr/bin/env python3

import os
import sys

from src.configuration import Configuration
from src.image_processing import ImageProcessor
from src.utils import print_error


def get_terminal_size() -> tuple:
    """Finds terminal size and returns it as a tuple (rows, columns)."""
    rows, columns = os.popen("stty size", "r").read().split()
    return int(rows), int(columns)


if __name__ == "__main__":
    CONFIG_LOCATION_PREFIX = "./"
    config = Configuration()
    if not config.read_and_apply_JSON_config(CONFIG_LOCATION_PREFIX + "config.json"):
        print_error("An error occurred when trying to apply config. Fallback to default config instead")
        if not config.read_and_apply_JSON_config(CONFIG_LOCATION_PREFIX + "default_config.json"):
            print_error("An error occurred when trying to apply default config. Can't operate")
            sys.exit(-1)

    target_file_path = "examples/input.png"
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

