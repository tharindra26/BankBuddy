from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.llms import Ollama
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from pinecone_manager import pinecone_manager
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.prompts import (
    ChatPromptTemplate
)

load_dotenv()

chat_history = []

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
    
def get_conversation_chain(vectorstore):
    # # llm = Ollama(model="llama3")
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
    # llm = ChatAnthropic(model='claude-3-opus-20240229')
    
    retriever = vectorstore.as_retriever(search_type="mmr")
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    prompt_with_history = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt_with_history)

    prompt_with_context = ChatPromptTemplate.from_messages([
        ("system", """
        
        You are a Commercial bank assistant who guide users on how to create accounts.

        Context: {context}
        
        """
        
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, prompt_with_context)

    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

    return retrieval_chain

def handle_userinput(question):
    retrieval_chain = get_conversation_chain(pinecone_manager.vectorDatabase)
    chat_history.append(HumanMessage(content=question))
    response = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": question
    })
    chat_history.append(AIMessage(content=response["answer"]))
    return response["answer"]