#!/usr/bin/env bash
set -xe

#https://www.glfw.org/docs/3.3/compile.html

# NOTE: I need to be run from the root of the repository
# and have all the dependencies install

cd vendor/glfw
cmake -S . -B build -DGLFW_BUILD_X11=ON -DGLFW_BUILD_WAYLAND=ON -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_DOCS=OFF
cmake --build build/
