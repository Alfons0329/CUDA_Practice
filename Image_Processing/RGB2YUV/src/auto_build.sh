#!/bin/bash
set -ex

rebuild=$1
if [ $rebuild -e 1 ]
then
    rm -rf ./build/*
fi

mkdir -p build && cd build
cmake ../
make -j16