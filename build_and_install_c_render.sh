#!/usr/bin/env bash

PYTHON_INTERPRETER_INCLUDE_DIR=/usr/include/python3.12
NUMPY_INCLUDE_DIR=venv/lib/python3.12/site-packages/numpy/_core/include
INSTALLATION_DIR=venv/lib/python3.12/site-packages

PYTHON_INTERPRETER_INCLUDE_DIR=$(realpath $PYTHON_INTERPRETER_INCLUDE_DIR)
NUMPY_INCLUDE_DIR=$(realpath $NUMPY_INCLUDE_DIR)
INSTALLATION_DIR=$(realpath $INSTALLATION_DIR)

cd src/c

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
