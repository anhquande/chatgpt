#!/usr/bin/env python3

import os
import sys
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

query = ' '.join(sys.argv[1:])

pdfLoader = PyPDFLoader("data/mycv.pdf")

textLoader = DirectoryLoader('./data',glob='*.txt')
index = VectorstoreIndexCreator().from_loaders([textLoader, pdfLoader])

print(index.query(query))