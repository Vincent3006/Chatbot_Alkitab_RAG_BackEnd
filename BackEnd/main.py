# main.py
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from fastapi.responses import StreamingResponse
import json
from rag_pipeline import initialize_rag_pipeline
from typing import List, Dict, Any
app = FastAPI(
    title="Chatbot Alkitab",
    description="Implementasi Semantic Chunking pada Arsitektur chatbot Alkitab berbasis RAG",
    version="1.0.0"
)
ALLOWED_ORIGINS = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ChatRequest(BaseModel):
    question: str
class SourceDocument(BaseModel):
    source_pdf: str
    source_range: str
class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
class RetrievedDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float

class RetrieveResponse(BaseModel):
    documents: List[RetrievedDocument]
@app.on_event("startup")
def startup_event():
    global custom_rag_pipeline
    print("Mulai Startup")
    custom_rag_pipeline = initialize_rag_pipeline()
@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_rag(request: ChatRequest):
    if not custom_rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG chain belum siap. Silakan coba lagi sesaat.")
    if not request.question:
        raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong.")
    try:
        result = custom_rag_pipeline.invoke(request.question)
        sources = []
        if result.get("source_documents"):
            for doc in result["source_documents"]:
                sources.append(SourceDocument(
                    source_pdf=doc.metadata.get('source_pdf', 'N/A'),
                    source_range=doc.metadata.get('source_range', 'N/A')
                ))
        return ChatResponse(answer=result["answer"], sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi error internal: {str(e)}")
@app.post("/api/retrieve", response_model=RetrieveResponse)
async def retrieve_only(request: ChatRequest):
    """
    Endpoint untuk hanya melakukan retrieval dokumen berdasarkan pertanyaan.
    """
    if not custom_rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG chain belum siap. Silakan coba lagi sesaat.")
    
    if not request.question:
        raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong.")

    try:
        # Panggil fungsi retrieve_documents yang baru dibuat
        retrieved_results = custom_rag_pipeline.retrieve_documents(request.question)
        
        # Kembalikan hasilnya dalam format RetrieveResponse
        return RetrieveResponse(documents=retrieved_results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi error internal: {str(e)}")
if __name__ == "__main__":
    print("Menjalankan server FastAPI di http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)