# Based on https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v1/

import pandas as pd
from llama_index.core import Document

import glob

txt_inputs = glob.glob('processed/plaintext/*/*.txt')
print(txt_inputs)


documents = []
for tex_file in txt_inputs:
    print('Adding %s'%tex_file)
    with open(tex_file) as f:
        documents.append(Document(text=f.read()))


from llama_index.llms.openai import OpenAI

#llm = OpenAI(model="gpt-4")

# one extra dep
from llama_index.core import VectorStoreIndex

# create an index from the parsed markdown
index = VectorStoreIndex.from_documents(documents)

# create a query engine for the index
query_engine = index.as_query_engine()

# query the engine
query = "What can you tell me about the neurons of the pharynx?"
response = query_engine.query(query)
print(response)
