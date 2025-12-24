"""
Quran & Hadith QA System - FastAPI Server
Run with: uvicorn main:app --reload
Or simply: python main.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import List, Optional
import uvicorn
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "./model_files"
FAISS_INDEX = f"{MODEL_PATH}/faiss_index.bin"
CHUNKED_DATA = f"{MODEL_PATH}/chunked_data.csv"
METADATA = f"{MODEL_PATH}/metadata.pkl"

# ============================================================================
# LOAD MODEL (Once at startup)
# ============================================================================

print("="*70)
print("🚀 Starting Quran & Hadith QA API Server")
print("="*70)

# Check if model files exist
if not os.path.exists(MODEL_PATH):
    print(f"\n❌ ERROR: '{MODEL_PATH}' folder not found!")
    print("Please create 'model_files' folder and add:")
    print("  • faiss_index.bin")
    print("  • chunked_data.csv")
    print("  • metadata.pkl")
    exit(1)

print("\n[1/4] Loading metadata...")
try:
    with open(METADATA, 'rb') as f:
        metadata = pickle.load(f)
    print(f"✓ Model: {metadata['model_name']}")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

print("\n[2/4] Loading FAISS index...")
try:
    index = faiss.read_index(FAISS_INDEX)
    print(f"✓ Loaded {index.ntotal:,} vectors")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

print("\n[3/4] Loading text database...")
try:
    chunked_df = pd.read_csv(CHUNKED_DATA)
    print(f"✓ Loaded {len(chunked_df):,} text chunks")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

print("\n[4/4] Loading embedding model (this may take 1-2 minutes)...")
try:
    embedding_model = SentenceTransformer(
        metadata['model_name'],
        device='cpu'  # CPU only
    )
    print(f"✓ Model loaded on CPU")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

print("\n" + "="*70)
print("✅ All components loaded successfully!")
print("="*70)

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Quran & Hadith QA API",
    description="Authentic Islamic Q&A powered by Quran and Hadith",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    min_score: Optional[float] = 0.3

class Source(BaseModel):
    id: int
    type: str
    reference: str
    text: str
    score: float

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]
    confidence: float
    total_retrieved: int

# ============================================================================
# CORE QA LOGIC
# ============================================================================

def retrieve_relevant_chunks(query: str, top_k: int = 5, min_score: float = 0.3):
    """Retrieve relevant text chunks using FAISS"""
    
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype('float32')
    
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for idx, score in zip(indices[0], distances[0]):
        if score >= min_score:
            result = chunked_df.iloc[idx]
            results.append({
                'text': result['text'],
                'source_type': result['source_type'],
                'language': result['language'],
                'reference': result['reference'],
                'score': float(score),
                'chunk_id': result['chunk_id']
            })
    
    return results

def generate_answer(query: str, top_k: int = 5, min_score: float = 0.3):
    """Generate complete answer with sources"""
    
    retrieved_chunks = retrieve_relevant_chunks(query, top_k, min_score)
    
    if not retrieved_chunks:
        return {
            'question': query,
            'answer': "I couldn't find relevant information for this query. Please try rephrasing.",
            'sources': [],
            'confidence': 0.0,
            'total_retrieved': 0
        }
    
    answer_parts = []
    sources = []
    
    for i, chunk in enumerate(retrieved_chunks[:3], 1):
        source_type = chunk['source_type'].title()
        reference = chunk['reference']
        text = chunk['text'][:400]
        
        source_info = f"[{i}] {source_type} - {reference}"
        sources.append({
            'id': i,
            'type': source_type,
            'reference': reference,
            'text': text,
            'score': round(chunk['score'], 4)
        })
        
        answer_parts.append(f"{source_info}:\n{text}\n")
    
    answer = "\n".join(answer_parts)
    avg_score = np.mean([c['score'] for c in retrieved_chunks[:3]])
    
    return {
        'question': query,
        'answer': answer,
        'sources': sources,
        'confidence': round(float(avg_score), 4),
        'total_retrieved': len(retrieved_chunks)
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend UI"""
    return FileResponse("index.html")

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about Islam"""
    
    try:
        if not request.question or len(request.question.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Question must be at least 3 characters long"
            )
        
        result = generate_answer(
            query=request.question,
            top_k=request.top_k,
            min_score=request.min_score
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )

@app.post("/batch_ask")
async def batch_questions(questions: List[str], top_k: int = 5):
    """Ask multiple questions at once"""
    
    try:
        if not questions or len(questions) == 0:
            raise HTTPException(
                status_code=400,
                detail="Provide at least one question"
            )
        
        if len(questions) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 questions per batch"
            )
        
        results = []
        for question in questions:
            result = generate_answer(question, top_k)
            results.append(result)
        
        return {
            "total_questions": len(questions),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "total_vectors": int(index.ntotal),
        "total_chunks": len(chunked_df),
        "embedding_dimension": metadata['embedding_dim'],
        "model_name": metadata['model_name'],
        "source_breakdown": {
            "hadith": int(metadata['dataset_stats']['hadith_count']),
            "quran": int(metadata['dataset_stats']['quran_count'])
        },
        "languages": metadata['dataset_stats']['languages']
    }

@app.get("/search")
async def search_texts(query: str, top_k: int = 10):
    """Simple search without answer generation"""
    
    try:
        if not query or len(query.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Query must be at least 3 characters"
            )
        
        results = retrieve_relevant_chunks(query, top_k, min_score=0.2)
        
        return {
            "query": query,
            "total_found": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌐 Starting FastAPI server...")
    print("="*70)
    print("\n📍 API will be available at:")
    print("   • Local:        http://localhost:8000")
    print("   • Your Network: http://0.0.0.0:8000")
    print("\n📚 Documentation:")
    print("   • Swagger UI: http://localhost:8000/docs")
    print("   • ReDoc:      http://localhost:8000/redoc")
    print("\n💡 Test with:")
    print("   python test_api.py")
    print("\n" + "="*70)
    print("\nPress Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )