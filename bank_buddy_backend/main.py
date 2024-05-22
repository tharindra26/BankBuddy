import os
from fastapi import FastAPI, UploadFile, File
from utils import get_pdf_text, get_text_chunks, initialize_pinecone, append_pinecone_vectorstore, get_pinecone_vectorstore, handle_userinput
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

class Question(BaseModel):
    user_question: str
    
pinecone_index = '' 
vectorDatabase = ''

@app.get("/")
def init():
    global pinecone_index
    global vectorDatabase  
    pinecone_index = initialize_pinecone()
    vectorDatabase = get_pinecone_vectorstore(pinecone_index)
    return {"message": pinecone_index}

@app.post("/chat/")
async def get_userinput(question: Question):
    print(question.user_question)
    answer = handle_userinput(question.user_question)
    print(answer)
    return {"answer": answer}

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        uploaded_files = []
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
            
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            
            uploaded_files.append(file_path)
        
        text = get_pdf_text(uploaded_files)
        text_chunks = get_text_chunks(text)
        append_pinecone_vectorstore(text_chunks, pinecone_index)

        return {"uploaded_files": uploaded_files, "text": text}
    
    finally:
        for file_path in uploaded_files:
            if os.path.exists(file_path):
                os.remove(file_path)