#!/bin/bash
set -ex

ruff format openworm_ai/*/*.py openworm_ai/*.py
ruff check openworm_ai/*/*.py openworm_ai/*.py

pip install .

python -m openworm_ai.quiz.QuizModel

python -m openworm_ai.parser.ParseWormAtlas


if [ $# -eq 1 ] ; then
    if [ $1 == "-free" ]; then
        python -m openworm_ai.graphrag.GraphRAG_test -test
    fi
    if [ $1 == "-llm" ]; then
        python -m openworm_ai.utils.llms -o-l32
        python -m openworm_ai.quiz.Templates -o-m
    fi
else 
        
    python -m openworm_ai.graphrag.GraphRAG_test

fi

echo
echo "  Success!"
echo
  

