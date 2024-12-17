#!/bin/bash
set -ex

ruff format openworm_ai/*/*.py openworm_ai/*.py
ruff check openworm_ai/*/*.py openworm_ai/*.py

pip install .

python -m openworm_ai.quiz.QuizModel

python -m openworm_ai.parser.ParseWormAtlas

python -m openworm_ai.graphrag.GraphRAG_test



echo
echo "  Success!"
echo
  

