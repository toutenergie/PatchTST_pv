#!/bin/bash
# Build and execute all notebooks of the PV study sequentially.
set -e
cd "$(dirname "$0")"
python3 nb00_eda.py
for k in 1 2 3 4; do python3 nb_config.py $k; done
python3 nb05_final.py
echo ALL_DONE
