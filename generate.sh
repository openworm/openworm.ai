#!/bin/bash
set -ex

ruff format *.py
ruff check *.py

python GraphRAG_test.py

echo
echo "  Success!"
echo
  

