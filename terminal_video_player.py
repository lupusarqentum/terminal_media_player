#!/usr/bin/env python3

import sys

from src.configuration import Configuration
from src.image_processing import ImageProcessor
from src.utils import print_error, print_warn, get_terminal_size

if __name__ == "__main__":
    CONFIG_LOCATION_PREFIX = "./"
    config = Configuration()
    if not config.read_and_apply_JSON_config(CONFIG_LOCATION_PREFIX + "config.json"):
        print_error("An error occurred when trying to apply config. Fallback to default config instead")
        if not config.read_and_apply_JSON_config(CONFIG_LOCATION_PREFIX + "default_config.json"):
            print_error("An error occurred when trying to apply default config. Can't operate")
            sys.exit(-1)

    if len(sys.argv) < 2:
        print_warn("No input file path was provided. Assuming examples/input.png")
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

