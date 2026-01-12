#!/bin/bash
# Pixi activation script to set up compiler environment

# Set clang as the default compiler
export CC="${CONDA_PREFIX}/bin/clang"
export CXX="${CONDA_PREFIX}/bin/clang++"
