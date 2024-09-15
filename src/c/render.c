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

static unsigned char max(unsigned char a, unsigned char b, unsigned char c) {
    if (a > b && a > c) {
        return a;
    }
    if (b > c) {
        return b;
    }
    return c;
}

static unsigned char min(unsigned char a, unsigned char b, unsigned char c) {
    if (a < b && a < c) {
        return a;
    }
    if (b < c) {
        return b;
    }
    return c;
}

static unsigned char shrink_channel(unsigned char channel) {
    if (channel <= 47) {
        return 0;
    }
    if (channel <= 115) {
        return 1;
    }
    if (channel <= 155) {
        return 2;
    }
    if (channel <= 195) {
        return 3;
    }
    if (channel <= 235) {
        return 4;
    }
    return 5;
}

static int get_color_number(unsigned char b, unsigned char g, unsigned char r) {
    if (max(b, g, r) - min(b, g, r) <= 10) {
        if (g <= 8) {
            return 232;
        }
        if (g >= 238) {
            return 255;
        }
        int upper = (18 - (int)g % 10) % 10 + g;
        int lower = upper - 10;
        if (upper - g < g - lower) {
            return 232 + (upper - 8) / 10;
        }
        return 232 + (lower - 8) / 10;
    }
    return 16 + 36 * shrink_channel(r) + 6 * shrink_channel(g) + shrink_channel(b);
}

char* TR_render(PyArrayObject* ascii_art, PyArrayObject* source_image, 
                    bool should_paint_back, bool should_paint_fore, 
                    bool boldify, unsigned char back_color_offset, 
                    unsigned char fore_color_offset, unsigned int terminal_columns) {
    TR_string* buffer = TR_create_string();
    unsigned char b, g, r;
    unsigned char b_, g_, r_;
    int color_number;
    unsigned int terminal_rows = PyArray_DIMS(ascii_art)[0];
    unsigned int actual_terminal_columns = PyArray_DIMS(ascii_art)[1];
    unsigned int offset_length = (terminal_columns - actual_terminal_columns) / 2;
    char* offset = PyMem_RawMalloc(offset_length + 1);
    memset(offset, ' ', offset_length);
    offset[offset_length] = '\0';
    
    if (boldify) {
        TR_append_cstring(buffer, "\033[1m");
    }
    
    /* TODO: reduce number of memory allocations by precalculating optimal initial capacity */
    
    for (unsigned int i = 0; i < terminal_rows; i++) {
        TR_append_cstring(buffer, offset);
        for (unsigned int j = 0; j < actual_terminal_columns; ++j) {
            if (should_paint_back || should_paint_fore) {
                unsigned char* color = (unsigned char*)PyArray_GETPTR2(source_image, i, j);
                b = color[0];
                g = color[1];
                r = color[2];
            }
            if (should_paint_back) {
                b_ = b + back_color_offset;
                g_ = g + back_color_offset;
                r_ = r + back_color_offset;
                color_number = get_color_number(b_, g_, r_);
                TR_append_cstring(buffer, "\033[48;5;");
                TR_append_number(buffer, color_number);
                TR_append_character(buffer, 'm');
            }
            if (should_paint_fore) {
                b_ = b + fore_color_offset;
                g_ = g + fore_color_offset;
                r_ = r + fore_color_offset;
                color_number = get_color_number(b_, g_, r_);
                TR_append_cstring(buffer, "\033[38;5;");
                TR_append_number(buffer, color_number);
                TR_append_character(buffer, 'm');
            }
            TR_append_character(buffer, ((char*)(PyArray_GETPTR2(ascii_art, i, j)))[0]);
        }
        if (should_paint_back) {
            TR_append_cstring(buffer, "\033[49m");
        }
        TR_append_character(buffer, '\n');
    }
    
    if (should_paint_fore) {
        TR_append_cstring(buffer, "\033[39m");
    }
    if (boldify) {
        TR_append_cstring(buffer, "\033[0m");
    }
    
    PyMem_RawFree(offset);
    return TR_shrink_and_free(buffer);
}

#endif
