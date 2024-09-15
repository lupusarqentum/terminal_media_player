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

#ifndef TR_RENDER_HEADER__
#define TR_RENDER_HEADER__

#include <numpy/arrayobject.h>
#include <stdbool.h>

/*
 * Note: caller must ensure that returned C-string is freed later.
 * Memory for resulting C-string will be allocated via
 *     Py_MemRawMalloc(size_t) function, hence it should be
 *     freed with Py_MemRawFree(void*).
 */

char* TR_render(PyArrayObject* grayscale, PyArrayObject* source_image, 
                PyArrayObject* intensity_to_grayscale,
                bool should_paint_back, bool should_paint_fore, 
                bool use_all_rgb, bool boldify, unsigned int terminal_columns);

/* This function should be called at least once before TR_render() call. */

void TR_init_color_tables(void);

#endif
