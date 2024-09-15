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

import numpy
import cv2
import terminalrenderer as tr

from src.python.configuration import Configuration


class ImageRenderer:
    """Renders a normal image into an ASCII art.

    Renders a normal image into a string of printable ASCII characters.
    The string might be directly printed into the terminal.
    """

    def __init__(self, config: Configuration) -> None:
        """Initializes with config containing rendering options."""
        self._character_aspect_ratio = config.get_character_aspect_ratio()
        self._boldify = config.get_boldify()
        self._paint_background = config.get_colorful_background_enabled()
        self._paint_foreground = config.get_colorful_charset_enabled()
        ascii_grayscale = config.get_ascii_characters_grayscale()
        self._generate_intensity_to_ascii_table(ascii_grayscale)

    def _generate_intensity_to_ascii_table(self, grayscale: str) -> None:
        def asciify_single_unit(intensity):
            pos = min(round(intensity / 255 * len(grayscale)),
                      len(grayscale) - 1)
            return grayscale[pos]
        table = [asciify_single_unit(intensity) for intensity in range(256)]
        self._intensity_to_ascii = numpy.array(table)

    def render(self, image: numpy.ndarray, terminal_rows: int,
               terminal_columns: int) -> None:
        """Renders an image.

        Number of rows and columns must be given
            to render ASCII art that will fit the terminal.

        Parameters:
            image: An image to render.
            terminal_rows: Number of rows in a rendered image.
            terminal_columns: Number of columns in a rendered image.

        Returns:
            A string that is a result of rendering.
            The string might be immediately printed or kept.
        """
        source_shape = image.shape
        new_size = find_ASCII_image_size(source_shape[0], source_shape[1],
                                         terminal_rows, terminal_columns,
                                         self._character_aspect_ratio)
        resized_image = cv2.resize(image, new_size)
        grayscale = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        rendered_frame = tr.render(grayscale,
                                   resized_image,
                                   self._intensity_to_ascii,
                                   self._paint_background,
                                   self._paint_foreground,
                                   self._boldify,
                                   terminal_columns)
        return rendered_frame


def find_ASCII_image_size(image_height: int, image_width: int,
                          terminal_rows: int, terminal_columns: int,
                          character_aspect_ratio: float) -> tuple:
    """Finds ASCII image size that fits terminal size and keeps aspect ratio.

    Finds appropriate (columns, rows) size to display ASCII art in terminal
        based on source image pixel (height, width) size.
    Aspect ratio of ASCII image will be the same as aspect ratio of source.
    To ensure keeping aspect ratio,
        correct value of terminal characters' aspect ratio is needed.

    Parameters:
        image_height: Height of source image in pixels.
        image_width: Width of source image in pixels.
        terminal_rows: Maximum allowed number of rows.
        terminal_columns: Maximum allowed number of columns.
        character_aspect_ratio: Aspect ratio of terminal characters
            (height / width)

    Returns:
        A tuple containing rows and columns for a new size.
    """
    new_aspect_ratio = image_height / (image_width * character_aspect_ratio)
    new_columns = min(terminal_columns, terminal_rows / new_aspect_ratio)
    new_rows = new_columns * new_aspect_ratio
    new_columns = round(new_columns)
    new_rows = round(new_rows)
    return (new_columns, new_rows)
