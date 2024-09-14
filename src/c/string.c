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

#ifndef TR_STRING_IMPL__
#define TR_STRING_IMPL__

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "string.h"

TR_string* TR_create_string() {
    TR_string* str = PyMem_RawMalloc(sizeof(TR_string));
    str->data = PyMem_RawMalloc(sizeof(char) * TR_STRING_INITIAL_CAPACITY);
    str->capacity = TR_STRING_INITIAL_CAPACITY;
    str->length = 0;
    return str;
}

void TR_append_character(TR_string* obj, char c) {
    if (obj->length == obj->capacity) {
        obj->capacity *= 2;
        PyMem_RawRealloc(obj->data, obj->capacity);
    }
    obj->data[obj->length++] = c;
}

void TR_append_cstring(TR_string* obj, char* s) {
    while ((*s) != '\0') {
        TR_append_character(obj, *s);
        s++;
    }
}

/* number must contain no more than three decimal digits */

void TR_append_number(TR_string* obj, int number) {
    if (100 <= number && number <= 999) {
        TR_append_character(obj, number / 100 + '0');
        number %= 100;
    }
    if (10 <= number && number <= 99) {
        TR_append_character(obj, number / 10 + '0');
        number %= 10;
    }
    TR_append_character(obj, number + '0');
}

/* Shrinks string's capacity to fit length, 
 *     appends null character,
 *     returns inner CString and frees string. 
 *     Caller must free returned pointer later. 
 */

char* TR_shrink_and_free(TR_string* obj) {
    TR_append_character(obj, '\0');
    PyMem_RawRealloc(obj->data, obj->length);
    char* result = obj->data;
    PyMem_RawFree(obj);
    return result;
}

#endif
