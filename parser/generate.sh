#!/bin/bash
set -ex

ruff format *.py
ruff check *.py

python ParseWormAtlas.py

echo
echo "  Success!"
echo
  

