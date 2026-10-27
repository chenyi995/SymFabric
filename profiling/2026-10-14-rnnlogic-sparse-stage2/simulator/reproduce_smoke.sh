#!/usr/bin/env bash
set -euo pipefail
SIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SIM_DIR"
test "$(git -C ramulator2 rev-parse HEAD)" = c6c7a5de0cef35935d6fad7369e6271406798865
cmake -S ramulator2 -B ramulator2/build -DCMAKE_BUILD_TYPE=Release -DRAMULATOR_PYTHON_BINDINGS=OFF > configure.log 2>&1
cmake --build ramulator2/build -j2 > build.log 2>&1
PYTHONPATH=ramulator2/python python3 -m ramulator export ramulator2/examples/example_config.py -o official_example.yaml
PYTHONPATH=ramulator2/python python3 -m ramulator export external_config.py -o external.yaml
g++ -std=c++20 -O2 -Iramulator2/src -Iramulator2/ext/fmt/include -Iramulator2/ext/yaml-cpp/include ramulator_driver.cpp -Lramulator2 -lramulator -Wl,-rpath,"$SIM_DIR/ramulator2" -o ramulator_driver
./ramulator_driver external.yaml external 1 1024 > external_serial.log 2>&1
./ramulator_driver external.yaml external 256 1024 > external_parallel.log 2>&1
(cd ramulator2 && ../ramulator_driver ../official_example.yaml example > ../official_example.log 2>&1)
git -C ramulator2 status --porcelain > upstream_clean.txt
test ! -s upstream_clean.txt
