from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Repository:
    def __init__(self):
        self.urls = [
          "https://lilianweng.github.io/posts/2023-06-23-agent/",
          "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
          "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]

    def add(self):
        docs = [WebBaseLoader(url).load() for url in self.urls]
        docs_list = [item for sublist in docs for item in sublist]
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
          chunk_size=250, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)
        # Adicionar ao vectorDB
        vectorstore = Chroma.from_documents(
          documents=doc_splits,
          collection_name="rag-chroma",
          embedding=OpenAIEmbeddings(),
          persist_directory="./chroma_db"  # pasta local onde será salvo
        )
        retriever = vectorstore.as_retriever()
        return retriever