# -*- coding: utf-8 -*-
"""
Created on Thu May  4 20:01:24 2023

@author: 21410
"""
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=bare-except
# pylint: disable=missing-class-docstring

import os

import pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CohereEmbeddings
# from langchain.llms import OpenAI
from langchain.llms import Cohere
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Pinecone

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Helper function for printing docs
def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

class MerlinBot:
    def __init__(self, config=None):
        # Initialize Pinecone
        self.config = config
        if self.config is None:
            self.config["COHERE_API_KEY"] = os.environ["COHERE_API_KEY"]
            self.config["PINECONE_API_KEY"] = os.environ["PINECONE_API_KEY"]
            self.config["PINECONE_API_ENV"] = os.environ["PINECONE_API_ENV"]
            self.config["PINECONE_INDEX_NAME"] = self.config["PINECONE_INDEX_NAME"]
        pinecone.init(
            api_key=self.config["PINECONE_API_KEY"],
            environment=self.config["PINECONE_API_ENV"]
        )
        self.vectorstore = None
        self.llm = None
        # self.embeddings = OpenAIEmbeddings(openai_api_key=self.config["OPENAI_API_KEY"])
        self.embeddings = CohereEmbeddings(cohere_api_key=self.config["COHERE_API_KEY"])
        self._init_llm()
        self._init_vectorstore()
        self._init_retriever_from_vectorstore()
    # def list_index(self):
    #     pass

    def create_new_index(self, index_name, dimension, **kwargs):
        return pinecone.create_index(index_name, dimension=dimension, **kwargs)

    def _delete_index(self, index_name):
        return pinecone.delete_index(index_name)

    def _update_pinecone_index(self, documents):
        Pinecone.from_documents(documents, self.embeddings, index_name=self.config["PINECONE_INDEX_NAME"])

    def _init_llm(self, temperature=0):
        # self.llm = OpenAI(temperature=temperature, openai_api_key=self.config["OPENAI_API_KEY"])
        self.llm = Cohere(temperature=temperature, cohere_api_key=self.config["COHERE_API_KEY"])

    def update_temperature(self, new_temperature):
        self.llm.temperature = new_temperature

    def _init_vectorstore(self):
        self.vectorstore = Pinecone.from_existing_index(index_name=self.config["PINECONE_INDEX_NAME"], embedding=self.embeddings)

    def _init_retriever_from_vectorstore(self):
        if self.vectorstore is None:
            self._init_vectorstore()
        self.retriever = self.vectorstore.as_retriever()

    def _load_pdfs(self, *args):
        loaded_pdf_files = []
        for pdf in list(args):
            try:
                loader = PyPDFLoader(pdf)
                document = loader.load()
                loaded_pdf_files.append(document)
            except:
                print(f"Error with argument {pdf}; please review and retry")
        return loaded_pdf_files

    def _split_documents(self, document_splitter, files):
        documents = []
        for file in files:
            documents += document_splitter.split_documents(file)
        return documents
    # update to public member
    def _split_documents_by_tokentext(self, files, chunk_size=1024, chunk_overlap=100, **kwargs):
        document_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        # documents = self._split_documents(document_splitter, files)
        documents = []
        for file in files:
            documents += document_splitter.split_documents(file)
        return documents

    def _init_conversation_bot(self):
        self._qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=memory
        )

    def perform_query(self, query):
        return self._qa_chain({"question":query})["answer"]
