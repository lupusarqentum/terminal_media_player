#!/usr/bin/env python3

import cv2
import numpy
import os

CHARACTER_ASPECT_RATIO = 2.25
TERMINAL_ROWS = None
TERMINAL_COLUMNS = None

ASCII_CHARACTERS_GRAYSCALE = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
#ASCII_CHARACTERS_GRAYSCALE = ".-:=!)[itCZ#3hX8%DRW"
POLARIZATION_LEVEL = 0.0

ENABLE_COLORFUL_BACK = True
ENABLE_COLORFUL_CHARSET = True
BACK_COLOR_OFFSET = 0
CHARSET_COLOR_OFFSET = 160

def find_ASCII_image_size(image_height, image_width, terminal_rows, terminal_columns, character_aspect_ratio):
    """Finds ASCII image size that fits terminal size and keeps pixel aspect ratio.
    
    Receives pixel size of an image and finds appropriate size to display ASCII art in terminal.
    Found ASCII image size is in rows and columns. 
    Number of terminal rows and columns is used to ensure that the size will fit the terminal.
    Aspect ratio (in pixels) of ASCII image with found size will be the same as aspect ratio of source image.
    
    Args:
        image_height: Height of source image in pixels.
        image_width: Width of source image in pixels.
        terminal_rows: Maximum allowed number of rows.
        terminal_columns: Maximum allowed number of columns.
        character_aspect_ratio: Aspect ratio of terminal characters (height / width)
    
    Returns:
        A tuple containing rows and columns for a new size.
    """
    
    #  Let's assume that terminal character width is 1 pixel; then the height is 
    #  character_aspect_ratio pixels.
    
    #  New_rows and new_columns must satisfy new_aspect_ratio = image_height / (image_width * character_aspect_ratio)
    #      Because if it's not satisfied, displayed ASCII image aspect ratio in PIXELS
    #      won't match source image aspect ratio.
    #  Hence, new_rows = new_columns * new_aspect_ratio.
    
    #  Let new_columns be X. Then either X = terminal_columns or X * new_aspect_ratio = terminal_rows
    #  and both X <= terminal_columns and X * new_aspect_ratio <= terminal_rows
    
    #  Now, it's obvious that X = min(terminal_columns, terminal_rows / new_aspect_ratio)
    
    new_aspect_ratio = image_height / (image_width * character_aspect_ratio)
    new_columns = min(terminal_columns, terminal_rows / new_aspect_ratio)
    new_rows = new_columns * new_aspect_ratio
    new_columns = round(new_columns)
    new_rows = round(new_rows)
    return (new_columns, new_rows)
    
    ratio_rows = old_rows / max_rows
    ratio_columns = old_columns / max_columns
    max_ratio = max(ratio_rows, ratio_columns)
    new_rows = old_rows / max_ratio
    new_columns = old_columns / max_ratio
    new_rows = max(round(new_rows), 1)
    new_columns = max(round(new_columns), 1)
    return (new_columns, new_rows)

def asciify_grayscale(grayscale):
    """Converts a grayscale of an image to ASCII image.
    
    Args:
        grayscale: A matrix of integers from 0 to 255 (gray intensity).
    
    Returns:
        A matrix of ASCII characters. 
    """
    def asciify_pixel(intensity):
        pos = min(round(intensity / 255 * len(ASCII_CHARACTERS_GRAYSCALE)), len(ASCII_CHARACTERS_GRAYSCALE) - 1)
        #return ASCII_CHARACTERS_GRAYSCALE[intensity]
        return ASCII_CHARACTERS_GRAYSCALE[pos]
    vectorized = numpy.vectorize(asciify_pixel)
    return vectorized(grayscale)

def polarize_grayscale(grayscale, polarization_level):
    """Pushes dark areas of an image darker and light areas lighter.
    
    Args:
        grayscale: A matrix of integers from 0 to 255 (gray intensity).
        polarization_level: A float number from 0.0 to 1.0 controlling how aggressively
            polarization should happen. For example, 0.0 value means nothing will happen at all,
            while 1.0 value means very aggressive polarization.
    
    Returns:
        A new matrix of integers from 0 to 255 (gray intensity of polarized grayscale).
    """
    if polarization_level == 0.0:
        return grayscale
    average_intensity = numpy.average(grayscale)
    multiplier = 127 / average_intensity * polarization_level
    return cv2.addWeighted(grayscale, multiplier, grayscale, 1.0 - polarization_level, 0.0)

def colorize_ascii_image(ascii_image, source_image):
    """Gives every ASCII image character a color.
    
    Assigns each ASCII image character one of the 240 colors via ANSI escape sequences.
    16 colors are not used because their value highly depends on the specific terminal.
    Assigned color is determined based on the color of the corresponding pixel in source_image.
    
    Args:
        ascii_image: A matrix of ASCII characters.
        source_image: An image. Must be the same shape as the ascii_image.
    
    Returns:
        A matrix containing python strings. Each string is a combination of escape sequences and ASCII character.
    """
    def identify_color(pixel):
        is_shade_of_gray = max(pixel) - min(pixel) < 15
        if is_shade_of_gray:
            return 232 + numpy.clip((int(pixel[0]) - 8) // 10, 0, 23)
        else:
            b = max(int(pixel[0]) - 95, -1) // 40 + 1
            g = max(int(pixel[1]) - 95, -1) // 40 + 1
            r = max(int(pixel[2]) - 95, -1) // 40 + 1
            return 16 + 36 * r + 6 * g + b
    result = numpy.zeros(ascii_image.shape, dtype="object")
    for i in range(ascii_image.shape[0]):
        for j in range(ascii_image.shape[1]):
            buffer = []
            if j == 0 and ENABLE_COLORFUL_CHARSET == False:
                buffer.append("\033[38;5;7m")
            elif ENABLE_COLORFUL_CHARSET:
                fore_color = str(identify_color(source_image[i, j] + CHARSET_COLOR_OFFSET))
                buffer.append("\033[38;5;" + fore_color + "m")
            if ENABLE_COLORFUL_BACK:
                back_color = str(identify_color(source_image[i, j] + BACK_COLOR_OFFSET))
                buffer.append("\033[48;5;" + back_color + "m")
            buffer.append(ascii_image[i, j])
            if j == ascii_image.shape[1] - 1:
                buffer.append("\033[39m");
                if ENABLE_COLORFUL_BACK:
                    buffer.append("\033[49m")
            result[i, j] = "".join(buffer)
    return result

def get_terminal_size():
    """Findss terminal size.
    
    Returns:
        a tuple of integers (rows, columns) -- terminal size.
    """
    rows, columns = os.popen("stty size", "r").read().split()
    return int(rows), int(columns)

def display_ascii_image(ascii_image, terminal_columns):
    """Prints an ASCII image to stdout centered horizontally.
    
    Args:
        ascii_image: An ASCII image to print. Must be a matrix of strings.
            Each string must contain exactly one printable character.
        terminal_columns: A size of the terminal in columns.
    
    Returns:
        None.
    """
    columns_used = ascii_image.shape[1]
    offset_length = (terminal_columns - columns_used) // 2
    offset = " " * offset_length
    for i in range(ascii_image.shape[0]):
        print(offset, end="")
        print("".join(ascii_image[i]))

input_file_name = "examples/input.png"
output_file_name = "examples/output.png"

TERMINAL_ROWS, TERMINAL_COLUMNS = get_terminal_size()

print('\033[49m')

input_image = cv2.imread(input_file_name)
new_size = find_ASCII_image_size(input_image.shape[0], input_image.shape[1], TERMINAL_ROWS, TERMINAL_COLUMNS, CHARACTER_ASPECT_RATIO)
resized_image = cv2.resize(input_image, new_size)
grayscale = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
polarized_grayscale = polarize_grayscale(grayscale, POLARIZATION_LEVEL)
ascii_image = asciify_grayscale(polarized_grayscale)
output_image = colorize_ascii_image(ascii_image, resized_image)
display_ascii_image(output_image, TERMINAL_COLUMNS)
