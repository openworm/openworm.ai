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
        doc = Document(text=f.read(), metadata={"file_name":tex_file})
        documents.append(doc)


from llama_index.llms.openai import OpenAI

#llm = OpenAI(model="gpt-4")

# one extra dep
from llama_index.core import VectorStoreIndex

# create an index from the parsed markdown
index = VectorStoreIndex.from_documents(documents)

# create a query engine for the index
query_engine = index.as_query_engine()

def process_query(response):
    response = query_engine.query(query)
    files_used = []
    for k in response.metadata:
        v = response.metadata[k]
        if 'file_name' in v:
            if not v['file_name'] in files_used:
                files_used.append(v['file_name'])

    print(f'''
===============================================================================
QUERY: {query}
-------------------------------------------------------------------------------
RESPONSE: {response}
SOURCES: {', '.join(files_used)}
===============================================================================
''')

# query the engine
query = "What can you tell me about the neurons of the pharynx of C. elegans?"
query = "Write 100 words on how C. elegans eats"
query = "How does the pharyngeal epithelium of C. elegans maintain its shape?"

queries = ["What can you tell me about the electrical connectivity between the muscles of C. elegans?", "What are the dimensions of the C. elegans pharynx?"]


for query in queries:
    process_query(query)
