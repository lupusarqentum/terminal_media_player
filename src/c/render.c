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

#ifndef TR_RENDER_IMPL__
#define TR_RENDER_IMPL__

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdbool.h>

#include "render.h"

char* TR_render(PyArrayObject* ascii_art, PyArrayObject* source_image, 
                    bool should_paint_back, bool should_paint_fore, 
                    bool boldify, unsigned char back_color_offset, 
                    unsigned char fore_color_offset, unsigned int terminal_columns) {
    char* result = PyMem_RawMalloc(14);
    result[0] =  'H';
    result[1] =  'x';
    result[2] =  'x';
    result[3] =  'x';
    result[4] =  'x';
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

#endif
