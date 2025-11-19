#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -d "vampire-4.9casc2024" ]; then
	rm -rf vampire-4.9casc2024
fi

unzip -q third-party/vampire/vampire-4.9casc2024.zip -d .

cd vampire-4.9casc2024

if [ -d "build" ]; then
	rm -rf build
fi

mkdir build && cd build

cmake ..
make

cd bin

vampire_binary=$(ls vampire_rel_main_* 2>/dev/null | head -1)
if [ -n "${vampire_binary}" ]; then
	mv "${vampire_binary}" vampire
elif [ ! -f "vampire" ]; then
	echo "Error: Vampire binary not found" >&2
	exit 1
fi

