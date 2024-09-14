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
#include "string.h"

char* TR_render(PyArrayObject* ascii_art, PyArrayObject* source_image, 
                    bool should_paint_back, bool should_paint_fore, 
                    bool boldify, unsigned char back_color_offset, 
                    unsigned char fore_color_offset, unsigned int terminal_columns) {
    TR_string* buffer = TR_create_string();
    TR_append_character(buffer, 'H');
    TR_append_character(buffer, 'e');
    TR_append_character(buffer, 'l');
    TR_append_character(buffer, 'l');
    TR_append_character(buffer, 'o');
    TR_append_character(buffer, ',');
    TR_append_character(buffer, ' ');
    TR_append_number(buffer, 342);
    TR_append_character(buffer, '@');
    TR_append_number(buffer, 28);
    TR_append_character(buffer, '#');
    TR_append_number(buffer, 7);
    char* result = TR_shrink_and_free(buffer);
    return result;
}

#endif
