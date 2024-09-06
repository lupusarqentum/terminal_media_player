#!/usr/bin/env python3

import cv2
import numpy

CHARACTER_ASPECT_RATIO = 2.25
TERMINAL_ROWS = 55
TERMINAL_COLUMNS = 238
ASCII_CHARACTERS_GRAYSCALE = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "

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
        return ASCII_CHARACTERS_GRAYSCALE[pos]
    vectorized = numpy.vectorize(asciify_pixel)
    return vectorized(grayscale)

input_file_name = "examples/input.png"
output_file_name = "examples/output.png"

input_image = cv2.imread(input_file_name)
new_size = find_ASCII_image_size(input_image.shape[0], input_image.shape[1], TERMINAL_ROWS, TERMINAL_COLUMNS, CHARACTER_ASPECT_RATIO)
resized_image = cv2.resize(input_image, new_size)
grayscale = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
output_image = asciify_grayscale(grayscale)
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        print(output_image[i, j], end="")
    print("")
cv2.imwrite(output_file_name, grayscale)
