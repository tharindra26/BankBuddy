from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.llms import Ollama
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pinecone_manager import pinecone_manager
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

load_dotenv()

memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

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
    # llm = Ollama(model="llama3")
    llm = ChatAnthropic(model='claude-3-opus-20240229')

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

    
    
    retriever=vectorstore.as_retriever()

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

    return conversation_chain

def handle_userinput(user_question):
    global chat_history 

    conversation_chain = get_conversation_chain(pinecone_manager.vectorDatabase)
    
    response = conversation_chain({"question": user_question})
    print(response)

    new_chat_history = response.get('chat_history', [])

    chat_history.extend(new_chat_history)

    ai_message_content = None
    for message in reversed(new_chat_history):
        if message.type == "ai":
            ai_message_content = message.content
            break


    return ai_message_content