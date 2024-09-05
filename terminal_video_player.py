#!/usr/bin/env python3

import cv2

def find_size_to_fit(old_rows, old_columns, max_rows, max_columns):
    """Finds new size of the image so that it fits given maximum size.
    
    Receives old size of an image and finds a new size, keeping rows/columns ratio. 
    No dimension in new size will be greater than the corresponding dimension in maximum size. 
    At least one dimension (rows or columns) in new size is equal to the maximum size.
    
    Args:
        old_rows: Number of rows in an old size.
        old_columns: Number of columns in an old size.
        max_rows: Maximum allowed number of rows.
        max_columns: Maximum allowed number of columns.
    
    Returns:
        A tuple containing rows and columns for a new size.
    """
    ratio_rows = old_rows / max_rows
    ratio_columns = old_columns / max_columns
    max_ratio = max(ratio_rows, ratio_columns)
    new_rows = old_rows / max_ratio
    new_columns = old_columns / max_ratio
    new_rows = max(round(new_rows), 1)
    new_columns = max(round(new_columns), 1)
    return (new_columns, new_rows)

input_file_name = "examples/input.png"
output_file_name = "examples/output.png"

input_image = cv2.imread(input_file_name)
new_size = find_size_to_fit(input_image.shape[0], input_image.shape[1], 55, 238)
output_image = cv2.resize(input_image, new_size)
cv2.imwrite(output_file_name, output_image)
