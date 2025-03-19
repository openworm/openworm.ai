#!/bin/bash
set -ex

ruff format openworm_ai/*/*.py openworm_ai/*.py
ruff check openworm_ai/*/*.py openworm_ai/*.py

pip install .

if [ $1 == "-quiz" ]; then
    python -m openworm_ai.quiz.QuizMaster 10
    python -m openworm_ai.quiz.QuizMaster -ask

elif [ $1 == "-llm" ]; then
    python -m openworm_ai.utils.llms -o-l32
    python -m openworm_ai.utils.llms -ge3
    python -m openworm_ai.quiz.Templates -o-m

else
    python -m openworm_ai.parser.DocumentModels
    python -m openworm_ai.quiz.QuizModel
    python -m openworm_ai.parser.ParseWormAtlas
    python -m openworm_ai.parser.ParseLlamaIndexJson

    if [ $1 == "-free" ]; then
        python -m openworm_ai.graphrag.GraphRAG_test -test
    else
        python -m openworm_ai.graphrag.GraphRAG_test $@

    fi
fi         

echo
echo "  Success!"
echo
  

