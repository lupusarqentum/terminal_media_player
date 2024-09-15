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

static unsigned char* BGR_to_8bit;

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

static unsigned char get_color_number(unsigned char b, unsigned char g, unsigned char r) {
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

char* TR_render(PyArrayObject* grayscale, PyArrayObject* source_image, 
                PyArrayObject* intensity_to_grayscale,
                bool should_paint_back, bool should_paint_fore, 
                bool boldify, unsigned int terminal_columns) {
    TR_string* buffer = TR_create_string();
    unsigned int terminal_rows = PyArray_DIMS(grayscale)[0];
    unsigned int actual_terminal_columns = PyArray_DIMS(grayscale)[1];
    unsigned int offset_length = (terminal_columns - actual_terminal_columns) / 2;
    char* offset = PyMem_RawMalloc(offset_length + 1);
    memset(offset, ' ', offset_length);
    offset[offset_length] = '\0';
    
    if (boldify) {
        TR_append_cstring(buffer, "\033[1m");
    }
    
    for (unsigned int i = 0; i < terminal_rows; i++) {
        TR_append_cstring(buffer, offset);
        for (unsigned int j = 0; j < actual_terminal_columns; ++j) {
            size_t index;
            if (should_paint_back || should_paint_fore) {
                unsigned char* color = (unsigned char*)PyArray_GETPTR2(source_image, i, j);
                index = (((size_t)color[0]) << 16) | 
                               (((size_t)color[1]) << 8) | ((size_t)color[2]);
            }
            if (should_paint_back) {
                TR_append_cstring(buffer, "\033[48;5;");
                TR_append_number(buffer, (int)BGR_to_8bit[index]);
                TR_append_character(buffer, 'm');
            }
            if (should_paint_fore) {
                TR_append_cstring(buffer, "\033[38;5;");
                TR_append_number(buffer, (int)BGR_to_8bit[index]);
                TR_append_character(buffer, 'm');
            }
            unsigned char intensity = ((unsigned char*)PyArray_GETPTR2(grayscale, i, j))[0];
            char ascii = ((char*)PyArray_GETPTR1(intensity_to_grayscale, intensity))[0];
            TR_append_character(buffer, ascii);
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

void TR_init_color_tables() {
    BGR_to_8bit = PyMem_RawMalloc(sizeof(unsigned char) * (1 << 24));
    for (size_t b = 0; b <= 255; b++) {
        for (size_t g = 0; g <= 255; g++) {
            for (size_t r = 0; r <= 255; r++) {
                size_t index = (b << 16) | (g << 8) | r;
                BGR_to_8bit[index] = get_color_number(b, g, r);
            }
        }
    }
}

#endif
