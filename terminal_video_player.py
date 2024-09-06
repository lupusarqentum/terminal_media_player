#!/usr/bin/env python3

import os

from src.image_processing import ImageProcessor
from src.configuration import Configuration


def get_terminal_size() -> tuple:
    """Finds terminal size.
    
    Returns:
        a tuple of integers (rows, columns) -- terminal size.
    """
    rows, columns = os.popen("stty size", "r").read().split()
    return int(rows), int(columns)


config = Configuration()
terminal_rows, terminal_columns = get_terminal_size()
image_processor = ImageProcessor(config)
image_processor.load(config.get_target_file_path())
image_processor.render(terminal_rows, terminal_columns)
image_processor.display()
