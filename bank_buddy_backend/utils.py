import os
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.llms import Ollama
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import ServerlessSpec, Pinecone as PC
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

load_dotenv()

pinecone_index = None
vectorDatabase = None

def handle_user_request(text, input_language, output_language):
    llm = ChatAnthropic(model='claude-3-opus-20240229')
    template = "You should translate a given {input_language} text to {output_language}."
    human_template = "{text}"
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])

    formatted_message = chat_prompt.format_messages(input_language=input_language, output_language=output_language, text=text)

    return llm.invoke(formatted_message)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def initialize_pinecone():
    global pinecone_index
    pc = PC(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    pinecone_index = index_name
    return index_name

def append_pinecone_vectorstore(text_chunks, index_name):
    embeddings = HuggingFaceEmbeddings()
    pinecone_vectorstore = Pinecone.from_texts(
        text_chunks, 
        embeddings, 
        index_name=index_name
    )
    return pinecone_vectorstore

def get_pinecone_vectorstore(index_name):
    global vectorDatabase
    try:
        embeddings = HuggingFaceEmbeddings()
        pinecone_vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        print("Vector store initialized successfully.")
        vectorDatabase = pinecone_vectorstore
        return pinecone_vectorstore
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return None
    
def get_conversation_chain(vectorstore):
    llm = Ollama(model="llama3")
    # llm = ChatAnthropic(model='claude-3-opus-20240229')

    system_template = """ 
    You are a helpful banking assistant for the Commercial Bank Of Ceylon PLC. Your goal is to assist users by providing accurate information based on the given data stored in Pinecone. If the user asks for PDFs, provide the corresponding links. If you don't know the answer, inform the user to contact or visit the nearest branch. Do not include the prompt itself in your responses. If the user is asking the question in sinhala languwage but using english letters, give the answer in sinhala letters.
    ----
    {context}
    ----
    """
    user_template = "Question:{question}"
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    retriever=vectorstore.as_retriever()

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

    return conversation_chain

initialize_pinecone()
get_pinecone_vectorstore(pinecone_index)

def handle_userinput(user_question):
    conversation_chain = get_conversation_chain(vectorDatabase)
    response = conversation_chain({"question": user_question})
    return response