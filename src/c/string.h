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

#ifndef TR_STRING_HEADER__
#define TR_STRING_HEADER__

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define TR_STRING_INITIAL_CAPACITY 16

typedef struct {
    char* data;
    size_t length;
    size_t capacity;
} TR_string;

TR_string* TR_create_string();

void TR_append_character(TR_string* obj, char c);

void TR_append_cstring(TR_string* obj, char* s);

/* number must contain no more than three decimal digits */

void TR_append_number(TR_string* obj, int number);

/* Shrinks string's capacity to fit length, 
 *     appends null character,
 *     returns inner CString and frees string. 
 *     Caller must free returned pointer later. 
 */

char* TR_shrink_and_free(TR_string* obj);

#endif
