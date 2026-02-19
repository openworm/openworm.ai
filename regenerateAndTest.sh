#!/bin/bash
set -ex

ruff format openworm_ai/*.py openworm_ai/*/*.py openworm_ai/*/*/*.py
ruff check openworm_ai/*.py openworm_ai/*/*.py openworm_ai/*/*/*.py

pip install .[dev]

if [ $1 == "-llamaparse" ]; then
    python -m openworm_ai.parser.llamaparse_backend 

elif [ $1 == "-quiz" ]; then
    python -m openworm_ai.quiz.QuizMaster 10
    python -m openworm_ai.quiz.QuizMaster -ask
    python -m openworm_ai.quiz.QuizMaster -ask -o-t

elif [ $1 == "-qplot" ]; then
    python -m openworm_ai.quiz.figures.quizplots_overcategories -nogui
    python -m openworm_ai.quiz.figures.quizplot_grid -nogui
    python -m openworm_ai.quiz.figures.quizplots -nogui

elif [ $1 == "-llm" ]; then

    python -m openworm_ai.utils.llms # default - ChatGPT via API
    python -m openworm_ai.utils.llms -co # Cohere via API - free
    python -m openworm_ai.utils.llms -g25 # gemini-2.5-flash via API - free tier
    python -m openworm_ai.utils.llms -o-l323b # Ollama:llama3.2:3b
    python -m openworm_ai.utils.llms -ge2 # Ollama:gemini2:latest
    python -m openworm_ai.utils.llms -o-qw # Ollama:qwen3:1.7b 


else
    python -m openworm_ai.parser.DocumentModels
    python -m openworm_ai.quiz.QuizModel
    python -m openworm_ai.quiz.figures.quizplot_grid -nogui
    python -m openworm_ai.parser.ParseWormAtlas

     #Do not call LlamaParse; use existing parsed outputs
    if [ $1 == "-free" ]; then
        python -m openworm_ai.parser.ParseLlamaIndexJson --skip
        python -m openworm_ai.graphrag.GraphRAG_test -test

    #Force full rebuild of raw/processed outputs
    elif [ $1 == "-reparse-all" ]; then
        python -m openworm_ai.parser.ParseLlamaIndexJson --reparse-all
        python -m openworm_ai.graphrag.GraphRAG_test $@
         
    #Default: incremental parse + monthly refresh (30 days)
    else
        python -m openworm_ai.parser.ParseLlamaIndexJson --max-age-days 30
        python -m openworm_ai.graphrag.GraphRAG_test $@

    fi
fi         

echo
echo "  Success!"
echo
  

