#!/bin/bash

#conan install . --output-folder external --build missing
meson setup builddir --prefix $PWD/NeoFOAM --cmake-prefix-path $PWD/NeoFOAM/ --libdir=lib
meson compile -C builddir
meson install -C builddir