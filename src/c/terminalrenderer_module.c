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
#include "render.h"

/**
 * Parameters list:
 *     grayscale
 *     source image
 *     intensity_to_grayscale
 *     should paint background?
 *     should paint foreground?
 *     should use full RGB space?
 *     should boldify foreground?
 *     terminal columns
 */

static PyObject* terminalrenderer_render(PyObject* self, PyObject* args) {
    PyArrayObject* grayscale;
    PyArrayObject* source_image;
    PyArrayObject* intensity_to_grayscale;
    bool should_paint_back;
    bool should_paint_fore;
    bool use_all_rgb;
    bool boldify;
    unsigned int terminal_columns;
    if (!PyArg_ParseTuple(args, "O!O!O!ppppI", 
                          &PyArray_Type, &grayscale,
                          &PyArray_Type, &source_image,
                          &PyArray_Type, &intensity_to_grayscale,
                          &should_paint_back, &should_paint_fore, 
                          &use_all_rgb, &boldify, &terminal_columns)) {
        return NULL;
    }
    
    int grayscale_art_ndims = PyArray_NDIM(grayscale);
    int source_image_ndims = PyArray_NDIM(source_image);
    int intensity_to_grayscale_ndims = PyArray_NDIM(intensity_to_grayscale);
    npy_intp* grayscale_dims = PyArray_DIMS(grayscale);
    npy_intp* source_image_dims = PyArray_DIMS(source_image);
    npy_intp* intensity_to_grayscale_dims = PyArray_DIMS(intensity_to_grayscale);
    
    if (intensity_to_grayscale_ndims != 1 || intensity_to_grayscale_dims[0] != 256 || PyArray_TYPE(intensity_to_grayscale) != NPY_UNICODE) {
        PyErr_SetString(PyExc_ValueError, "Intensity to ASCII translation array is invalid");
        return NULL;
    }
    if (grayscale_art_ndims != 2 || source_image_ndims != 3 || 
        grayscale_dims[0] != source_image_dims[0] || 
        grayscale_dims[1] != source_image_dims[1] || 
        source_image_dims[2] != 3) {
        PyErr_SetString(PyExc_ValueError, "Grayscale image shape or source image shape are invalid or mismatch");
        return NULL;
    }
    if (terminal_columns < source_image_dims[1]) {
        PyErr_SetString(PyExc_ValueError, "Terminal doesn't have enough columns to display an image");
        return NULL;
    }
    if (PyArray_TYPE(grayscale) != NPY_UBYTE || 
        PyArray_TYPE(source_image) != NPY_UBYTE) {
        PyErr_SetString(PyExc_TypeError, "Expected that elements of grayscale image and source image are uint8");
        return NULL;
    }
    
    char* result = TR_render(grayscale, source_image, intensity_to_grayscale,
                             should_paint_back, should_paint_fore, use_all_rgb,
                             boldify, terminal_columns);
    PyObject* pyResult = PyUnicode_FromString(result);
    PyMem_RawFree(result);
    return pyResult;
}

static PyMethodDef TerminalRendererMethods[] = {
    {"render", terminalrenderer_render, METH_VARARGS,
    "Generates ASCII art and returns it as a str object.\n\n"
    "Returned value can be immediately printed to stdout to display generated\n"
    "art. Optional painting and making characters bold is done via ANSI escape\n"
    "sequences. If characters are painted, at least 240 colors from ANSI escape\n"
    "sequences are used. Optionally might use full RGB space.\n\n"
    "Parameters:\n"
    "\tgrayscale: 2D numpy.ndarray. grayscaled source image of uint8.\n"
    "\tsource_image: 3D numpy.ndarray. Its shape must be (x, y, 3)\n"
    "\t\twhere (x, y) is the shape of grayscale.\n"
    "\tintensity_to_grayscale: 1D numpy.ndarray. Must contain 256 ASCII characters.\n"
    "\tshould_paint_background\n\tshould_paint_foreground\n"
    "\tuse_all_rgb: whether should use all RGB colors or just 240.\n"
    "\t\tMatters only if either should_paint_background or\n"
    "\t\tshould_paint_foreground is set.\n"
    "\tboldify: whether or not should make characters bold.\n"
    "\tterminal_columns: Number of columns in the terminal.\n"
    "\t\tMust not be less than height of source image.\n"
    "\t\tHeight of the generated art will match the size of source image.\n"
    "\t\tThis parameter is used to center generated ASCII art\n"
    "\t\tby adding spaces-offset to every line.\n"},
    {NULL, NULL, 0, NULL}
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
    TR_init_color_tables();
    return PyModule_Create(&terminalrenderermodule);
}
