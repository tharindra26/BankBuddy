import os
from dotenv import load_dotenv
from pinecone import Pinecone as PC, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone

load_dotenv()

class PineconeManager:
    def __init__(self):
        self.pinecone_index = None
        self.vectorDatabase = None

    def initialize_pinecone(self):
        pc = PC(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME")

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=768,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        self.pinecone_index = index_name
        return index_name

    def get_pinecone_vectorstore(self):
        try:
            embeddings = HuggingFaceEmbeddings()
            pinecone_vectorstore = PineconeVectorStore(index_name=self.pinecone_index, embedding=embeddings)
            self.vectorDatabase = pinecone_vectorstore
            print("Vector store initialized successfully.")
            return pinecone_vectorstore
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            return None

    def append_pinecone_vectorstore(self, text_chunks):
        embeddings = HuggingFaceEmbeddings()
        pinecone_vectorstore = Pinecone.from_texts(
            text_chunks, 
            embeddings, 
            index_name=self.pinecone_index
        )
        return pinecone_vectorstore

pinecone_manager = PineconeManager()
