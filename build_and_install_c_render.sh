#!/usr/bin/env bash

source dirs.txt

PYTHON_INTERPRETER_INCLUDE_DIR=$(realpath "$PYTHON_INTERPRETER_INCLUDE_DIR")
NUMPY_INCLUDE_DIR=$(realpath "$NUMPY_INCLUDE_DIR")
INSTALLATION_DIR=$(realpath "$INSTALLATION_DIR")

cd src/c || exit

echo Compiling extension module...

gcc -fPIC -c -I"$PYTHON_INTERPRETER_INCLUDE_DIR" -I"$NUMPY_INCLUDE_DIR" terminalrenderer_module.c render.c string.c

echo Linking shared library...

gcc --shared terminalrenderer_module.o render.o string.o -o terminalrenderer.so

echo Installing...

cp terminalrenderer.so "$INSTALLATION_DIR"

echo Cleaning...

rm -f terminalrenderer.so
rm -f terminalrenderer_module.o
rm -f render.o
rm -f string.o
