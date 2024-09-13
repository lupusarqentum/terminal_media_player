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
    Provide a configuration object containing rendering preferences to __init__
        to alter rendering process.
    """

    def __init__(self, config: Configuration) -> None:
        """Initializes image renderer with configuration object."""
        self._config = config

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
        character_aspect_ratio = self._config.get_character_aspect_ratio()
        polarization_level = self._config.get_polarization_level()
        ascii_grayscale = self._config.get_ascii_characters_grayscale()
        boldify = self._config.get_boldify()
        source_shape = image.shape

        paint_background = self._config.get_colorful_background_enabled()
        paint_foreground = self._config.get_colorful_charset_enabled()
        boldify_foreground = self._config.get_boldify()
        background_color_offset = self._config.get_background_color_offset()
        foreground_color_offset = self._config.get_charset_color_offset()

        new_size = find_ASCII_image_size(source_shape[0], source_shape[1],
                                         terminal_rows, terminal_columns,
                                         character_aspect_ratio)
        resized_image = cv2.resize(image, new_size)
        grayscale = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        polarized_grayscale = polarize_grayscale(grayscale, polarization_level)
        ascii_image = asciify_grayscale(polarized_grayscale, ascii_grayscale)

        return tr.render(ascii_image, resized_image, paint_background,
                         paint_foreground, boldify_foreground,
                         background_color_offset,
                         foreground_color_offset, terminal_columns)


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


def asciify_grayscale(grayscale: numpy.ndarray,
                      ascii_grayscale: str) -> numpy.ndarray:
    """Converts a grayscale of an image to ASCII image."""
    grayscale_len = len(ascii_grayscale)

    def asciify_pixel(intensity):
        pos = min(round(intensity / 255 * grayscale_len), grayscale_len - 1)
        return ascii_grayscale[pos]
    vectorized = numpy.vectorize(asciify_pixel)
    return vectorized(grayscale)


def polarize_grayscale(grayscale: numpy.ndarray,
                       polarization_level: float) -> numpy.ndarray:
    """Pushes dark areas of an image darker and light areas lighter.

    Parameters:
        grayscale: A matrix of integers from 0 to 255 (gray intensity).
        polarization_level: A float number from 0.0 to 1.0
            controlling how aggressively polarization should be performed.
            For example, 0.0 value means nothing will happen at all,
            while 1.0 value means very aggressive polarization.
    """
    if polarization_level == 0.0:
        return grayscale
    average_intensity = numpy.average(grayscale)
    multiplier = 127 / average_intensity * polarization_level
    return cv2.addWeighted(grayscale, multiplier,
                           grayscale, 1.0 - polarization_level, 0.0)
