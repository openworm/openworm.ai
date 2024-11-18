# Based on https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v1/

import pandas as pd
from llama_index.core import Document
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import load_index_from_storage

import glob

from modelspec.utils import load_json

STORE_DIR = "store"
SOURCE_DOCUMENT = "source document"

json_inputs = glob.glob('processed/json/*/*.json')
print(json_inputs)

documents = []
for json_file in json_inputs:
    print('Adding %s'%json_file)
    doc_model = load_json(json_file)
    for title in doc_model:
        print('  Processing document: %s'%title)
        doc_contents = doc_model[title]
        src_page = doc_contents['source']
        for section in doc_contents['sections']:
            print('    Processing section: %s'%doc_contents['sections'][section])
            all_text = ''
            if 'paragraphs' in doc_contents['sections'][section]:
                for p in doc_contents['sections'][section]['paragraphs']:
                    all_text += p['contents']+'\n\n'
            if len(all_text)==0:
                all_text = ' '
            #print(f'---------------------\n{all_text}\n---------------------')
            src_info = f'WormAtlas Handbook: [{title}, Section {section}]({src_page})'
            doc = Document(text=all_text, metadata={SOURCE_DOCUMENT:src_info})
            documents.append(doc)


from llama_index.llms.openai import OpenAI

#llm = OpenAI(model="gpt-4")

# one extra dep
from llama_index.core import VectorStoreIndex

# create an index from the parsed markdown
index = VectorStoreIndex.from_documents(documents)

index.storage_context.persist(persist_dir=STORE_DIR)

#index_reloaded =SimpleIndexStore.from_persist_dir(persist_dir=INDEX_STORE_DIR)
storage_context = StorageContext.from_defaults(
    docstore=SimpleDocumentStore.from_persist_dir(persist_dir=STORE_DIR),
    vector_store=SimpleVectorStore.from_persist_dir(
        persist_dir=STORE_DIR),
    index_store=SimpleIndexStore.from_persist_dir(persist_dir=STORE_DIR),
)
index_reloaded = load_index_from_storage(storage_context)

# create a query engine for the index
query_engine = index_reloaded.as_query_engine()

def process_query(response):
    response = query_engine.query(query)
    files_used = []
    for k in response.metadata:
        v = response.metadata[k]
        if SOURCE_DOCUMENT in v:
            if not v[SOURCE_DOCUMENT] in files_used:
                files_used.append(v[SOURCE_DOCUMENT])

    print(f'''
===============================================================================
QUERY: {query}
-------------------------------------------------------------------------------
RESPONSE: {response}
SOURCES: 
{',\n'.join(files_used)}
===============================================================================
''')

# query the engine
query = "What can you tell me about the neurons of the pharynx of C. elegans?"
query = "Write 100 words on how C. elegans eats"
query = "How does the pharyngeal epithelium of C. elegans maintain its shape?"

queries = ["What can you tell me about the properties of electrical connectivity between the muscles of C. elegans?", 
           "What are the dimensions of the C. elegans pharynx?", 'What color is C. elegans?']

for query in queries:
    process_query(query)
