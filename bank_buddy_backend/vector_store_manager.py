from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

class VectorStoreManager:
    def __init__(self):
        self.vectorDatabase = None

    def get_vectorstore(self):
        try:
            embeddings = HuggingFaceEmbeddings()
            vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
            self.vectorDatabase = vectorstore
            print("Vector store initialized successfully.")
            return vectorstore
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            return None

    def append_vectorstore(self, text_chunks):
        embeddings = HuggingFaceEmbeddings()
        vectorstore = Chroma.from_documents(text_chunks, embeddings, persist_directory="./chroma_db")
        return vectorstore

vector_store_manager = VectorStoreManager()