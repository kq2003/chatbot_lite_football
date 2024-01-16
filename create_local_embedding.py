from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import os
os.environ['OPENAI_API_KEY'] = 'my key'
openai_api_key = os.getenv("OPENAI_API_KEY", "my key")

loader = DirectoryLoader('corpus_new')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

persist_directory = 'embedding'

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
docsearch.persist()
