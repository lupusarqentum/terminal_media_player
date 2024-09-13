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
    char* result = "Hello, World!\n";
    PyObject* pyResult = PyUnicode_FromString(result);
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
    return PyModule_Create(&terminalrenderermodule);
}
