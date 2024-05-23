import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from pinecone_manager import pinecone_manager
from utils import get_pdf_text, get_text_chunks, handle_userinput

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

@app.on_event("startup")
async def startup_event():
    pinecone_manager.initialize_pinecone()
    pinecone_manager.get_pinecone_vectorstore()

@app.get("/")
def read_root():
    return {"message": "Server is running"}

@app.post("/chat/")
async def get_userinput(question: Question):
    print(question.user_question)
    user_question = question.user_question
    answer = handle_userinput(user_question)
    return {"user_question": user_question, "answer": answer}

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    uploaded_files = []
    try:
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
            
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            
            uploaded_files.append(file_path)
        
        text = get_pdf_text(uploaded_files)
        text_chunks = get_text_chunks(text)
        pinecone_manager.append_pinecone_vectorstore(text_chunks)

        return {"uploaded_files": uploaded_files, "text": text}
    
    finally:
        for file_path in uploaded_files:
            if os.path.exists(file_path):
                os.remove(file_path)




