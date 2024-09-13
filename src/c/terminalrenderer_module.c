/**
 * TerminalVideoPlayer, a program using command line interface to play videos.
 * Copyright (C) 2024  Roman Lisov
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see
 * https://www.gnu.org/licenses/gpl-3.0.html 
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdbool.h>

static char* render(PyArrayObject* ascii_art, PyArrayObject* source_image, 
                    bool should_paint_back, bool should_paint_fore, 
                    bool boldify, unsigned char back_color_offset, 
                    unsigned char fore_color_offset, unsigned int terminal_columns) {
    char* result = PyMem_RawMalloc(14); /* See also: PyMem_RawMalloc, PyMem_RawRealloc, PyMem_RawFree */
    result[0] =  'H';
    result[1] =  'e';
    result[2] =  'l';
    result[3] =  'l';
    result[4] =  'o';
    result[5] =  ',';
    result[6] =  ' ';
    result[7] =  'W';
    result[8] =  'o';
    result[9] =  'r';
    result[10] = 'l';
    result[11] = 'd';
    result[12] = '!';
    result[13] = '\0';
    return result;
}

/**
 * Parameters list:
 *     ascii_art
 *     source_image
 *     paint_background?
 *     paint_foreground?
 *     boldify_foreground?
 *     background_color_offset
 *     foreground_color_offset
 *     terminal_columns
 */

static PyObject* terminalrenderer_render(PyObject* self, PyObject* args) {
    PyArrayObject* ascii_art;
    PyArrayObject* source_image;
    bool should_paint_back;
    bool should_paint_fore;
    bool boldify;
    unsigned char back_color_offset;
    unsigned char fore_color_offset;
    unsigned int terminal_columns;
    if (!PyArg_ParseTuple(args, "O!O!pppbbI", 
                          &PyArray_Type, &ascii_art,
                          &PyArray_Type, &source_image,
                          &should_paint_back, &should_paint_fore,
                          &boldify, &back_color_offset,
                          &fore_color_offset, &terminal_columns)) {
        return NULL;
    }
    
    int ascii_art_ndims = PyArray_NDIM(ascii_art);
    int source_image_ndims = PyArray_NDIM(source_image);
    npy_intp* ascii_art_dims = PyArray_DIMS(ascii_art);
    npy_intp* source_image_dims = PyArray_DIMS(source_image);
    
    if (ascii_art_ndims != 2 || source_image_ndims != 3 || 
        ascii_art_dims[0] != source_image_dims[0] || 
        ascii_art_dims[1] != source_image_dims[1] || 
        source_image_dims[2] != 3) {
        PyErr_SetString(PyExc_ValueError, "ASCII art shape or source image shape are invalid or mismatch");
        return NULL;
    }
    if (terminal_columns < source_image_dims[1]) {
        PyErr_SetString(PyExc_ValueError, "Terminal doesn't have enough columns to display an image");
        return NULL;
    }
    if (PyArray_TYPE(ascii_art) != NPY_UNICODE || 
        PyArray_TYPE(source_image) != NPY_UBYTE) {
        PyErr_SetString(PyExc_TypeError, "Expected that an element of ascii_art is of Unicode type and an element of source_image is uint8");
        return NULL;
    }
    
    char* result = render(ascii_art, source_image, should_paint_back, 
                          should_paint_fore, boldify, back_color_offset, 
                          fore_color_offset, terminal_columns);
    PyObject* pyResult = PyUnicode_FromString(result);
    PyMem_RawFree(result);
    return pyResult;
}

static PyMethodDef TerminalRendererMethods[] = {
    {"render", terminalrenderer_render, METH_VARARGS, 
    "Paints an ASCII art and returns result as a str object.\n\nReceives an ASCII art and its colorful source image, paints characters\nand background if needed, and returns the result as a str object\nthat can be immediately printed to stdout. Painting and, optionally,\nmaking characters bold, is done via ANSI escape sequences. Only 240 colors\nof 8-bit terminal colors are used.\n\nParameters:\n\tascii_art: 2D numpy.ndarray of ASCII characters.\n\tsource_image: Colorful image that was used to produce ascii_art.\n\t\tIf source_image's width and height are x and y,\n\t\tascii_art's width and height must be x and y.\n\tpaint_background: True, if should paint background, False, otherwise.\n\tpaint_foreground: True, if should paint characters, False otherwise.\n\tboldify_foreground: True, if should make characters bold, False otherwise.\n\tbackground_color_offset: Unsigned 8-bit integer that will be addedn\n\t\tto all color channel of source_image\n\t\tbefore determining background color.\n\tforeground_color_offset: Same thing applied to characters.\n\tterminal_columns: Number of columns in a terminal in which a\n\t\tresulting image will be printed."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef terminalrenderermodule = {
    PyModuleDef_HEAD_INIT,
    "terminalrenderer",
    NULL,
    -1,
    TerminalRendererMethods
};

PyMODINIT_FUNC PyInit_terminalrenderer(void) {
    import_array();
    return PyModule_Create(&terminalrenderermodule);
}
