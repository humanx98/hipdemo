#!/bin/bash

set -e

script_dir=$(dirname $0)
root_dir=$(realpath "$script_dir/../..")

cd $root_dir/submodules/hiprt
cmake -DCMAKE_BUILD_TYPE=Release -DBITCODE=ON -DNO_ENCRYPT=ON -S . -B build
cmake --build build --config Release
cd scripts/bitcodes

python compile.py
cmake -DCMAKE_BUILD_TYPE=Debug -DBITCODE=ON -DNO_ENCRYPT=ON -S . -B build/debug
cmake --build build/debug --config Debug