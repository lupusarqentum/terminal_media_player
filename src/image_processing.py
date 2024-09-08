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

from src.configuration import Configuration


class ImageProcessor:
    """Processor capable of rendering and displaying image.

    Displaying an image typically consists of three steps.
    Firstly, any image must be loaded into memory from file.
    Secondly, an image must be rendered into ASCII characters.
    Thirdly, a rendered image must be printed to the stdout.
    Each step is represented by a separate method.
    Each method might be called several times if needed.
    For example, you might re-call render()
        to rerender image with different parameters.
    For another one, you could call display() method several times
        to display an image several times even if it was rendered just once.
    """

    def __init__(self, config: Configuration) -> None:
        """Initializes self.

        Initializes image processor by passing configuration object.
        Config is used to alter rendering process by configuring it.
        """
        self._config = config
        self._hasLoaded = False
        self._hasRendered = False

    def load(self, target_file_path: str) -> None:
        """Receives file path to the target image and loads it into memory.

        Raises:
            IOError: If failed to read an image.
        """
        self._source_image = cv2.imread(target_file_path)
        if self._source_image is None:
            raise IOError("Failed to read an image: " + target_file_path)
        self._hasLoaded = True
        self._hasRendered = False

    def render(self, terminal_rows: int, terminal_columns: int) -> None:
        """Renders an image.

        Renders an image that was loaded with the load method.
        Image will be rendered, but not displayed.
        Number of rows and columns must be given
            to render image with correct size.

        Parameters:
            terminal_rows: Number of rows in a rendered image.
            terminal_columns: Number of columns in a rendered image.

        Raises:
            ValueError: If no image was loaded with the load method.
        """
        if not self._hasLoaded:
            raise ValueError("Can't render an image: no image was loaded")

        character_aspect_ratio = self._config.get_character_aspect_ratio()
        polarization_level = self._config.get_polarization_level()
        ascii_grayscale = self._config.get_ascii_characters_grayscale()
        charset_offset = self._config.get_charset_color_offset()
        background_offset = self._config.get_background_color_offset()
        source_shape = self._source_image.shape

        new_size = find_ASCII_image_size(source_shape[0], source_shape[1],
                                         terminal_rows, terminal_columns,
                                         character_aspect_ratio)

        resized_image = cv2.resize(self._source_image, new_size)
        grayscale = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        polarized_grayscale = polarize_grayscale(grayscale, polarization_level)
        ascii_image = asciify_grayscale(polarized_grayscale, ascii_grayscale)
        if self._config.get_colorful_charset_enabled():
            ascii_image = colorize_ascii_image_characters(ascii_image,
                                                          resized_image,
                                                          charset_offset)
        if self._config.get_colorful_background_enabled():
            ascii_image = colorize_ascii_image_background(ascii_image,
                                                          resized_image,
                                                          background_offset)
        self._rendered_image = ascii_image
        self._terminal_columns = terminal_columns
        self._hasRendered = True

    def display(self) -> None:
        """Displays rendered image.

        Raises:
            ValueError: If no image was previously rendered.
        """
        if not self._hasRendered:
            raise ValueError("Can't display an image: no image was rendered")
        display_ascii_image(self._rendered_image, self._terminal_columns)


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


def find_escape_sequence(color: list, for_character: bool) -> str:
    """Finds ANSI escape sequence best matching given BGR color.

    Finds one of 240 ANSI escape sequences that sets the best matching color.
    16 possible colors out of all 256 are not used
        because their exact BGR values depend on a specific terminal used.

    Parameters:
        color: A color.
        for_character: True, if need to set color for foreground,
            and False if need to set color for background.

    Returns:
        An ANSI escape sequence.
    """
    is_shade_of_gray = max(color) - min(color) < 15
    color_number = None
    if is_shade_of_gray:
        color_number = 232 + numpy.clip((int(color[0]) - 8) // 10, 0, 23)
    else:
        b = max(int(color[0]) - 95, -1) // 40 + 1
        g = max(int(color[1]) - 95, -1) // 40 + 1
        r = max(int(color[2]) - 95, -1) // 40 + 1
        color_number = 16 + 36 * r + 6 * g + b
    beginning = "\033[38;5;" if for_character else "\033[48;5;"
    return beginning + str(color_number) + "m"


def colorize_ascii_image_characters(ascii_image: numpy.ndarray,
                                    source_image: numpy.ndarray,
                                    color_offset: int) -> numpy.ndarray:
    """Gives every ASCII image character a foreground color.

    Assigns each ASCII image character foreground color
        based on source_image color.

    Parameters:
        ascii_image: A matrix of ASCII characters.
        source_image: An image. Must be the same shape as the ascii_image.
        color_offset: A value to add to every color channel of foreground.

    Returns:
        An array of strings.
            Each one is combination of escape sequences and a character.
    """
    result = numpy.zeros(ascii_image.shape, dtype="object")
    source_offset = source_image + color_offset
    for i in range(ascii_image.shape[0]):
        for j in range(ascii_image.shape[1]):
            color_sequence = find_escape_sequence(source_offset[i, j], True)
            result[i, j] = color_sequence + ascii_image[i, j]
    result[ascii_image.shape[0] - 1, ascii_image.shape[1] - 1] += "\033[39m"
    return result


def colorize_ascii_image_background(ascii_image: numpy.ndarray,
                                    source_image: numpy.ndarray,
                                    color_offset: int) -> numpy.ndarray:
    """Gives every ASCII image character a background color.

    Assigns each ASCII image character backround color
        based on source_image color.

    Parameters:
        ascii_image: A matrix of ASCII characters.
        source_image: An image. Must be the same shape as the ascii_image.
        color_offset: A value to add to every color channel of background.

    Returns:
        An array of strings.
            Each one is combination of escape sequences and a character.
    """
    result = numpy.zeros(ascii_image.shape, dtype="object")
    source_offset = source_image + color_offset
    for i in range(ascii_image.shape[0]):
        for j in range(ascii_image.shape[1]):
            color_sequence = find_escape_sequence(source_offset[i, j], False)
            result[i, j] = color_sequence + ascii_image[i, j]
    for i in range(ascii_image.shape[0]):
        result[i, ascii_image.shape[1] - 1] += "\033[49m"
    return result


def display_ascii_image(ascii_image: numpy.ndarray,
                        terminal_columns: int) -> None:
    """Prints an ASCII image to stdout centered horizontally.

    Parameters:
        ascii_image: An ASCII image to print. Must be a matrix of strings.
            Each string must contain exactly one printable character.
        terminal_columns: A size of the terminal in columns.
    """
    columns_used = ascii_image.shape[1]
    offset_length = (terminal_columns - columns_used) // 2
    offset = " " * offset_length
    for i in range(ascii_image.shape[0]):
        print(offset, end="")
        print("".join(ascii_image[i]))
