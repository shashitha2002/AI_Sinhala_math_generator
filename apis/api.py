"""
FastAPI Application for Sinhala Mathematics Question Generation
Uses the SinhalaRAGSystem from rag_model.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
import os
import time
from dotenv import load_dotenv

# Import the RAG model
from models.rag_model import SinhalaRAGSystem

load_dotenv()

# Configuration
GEMINI_API_KEY = os. getenv("GEMINI_API_KEY", "")

# Initialize FastAPI app
app = FastAPI(
    title="Sinhala Math Question Generator API",
    description="Generate O/L Sinhala math questions using RAG + Google Gemini API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
rag_system:  Optional[SinhalaRAGSystem] = None
system_status = {
    "initialized": False,
    "model_name": None,
    "last_error": None,
    "data_loaded": False
}


# ==================== Pydantic Models ====================

class DifficultyLevel(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class QuestionRequest(BaseModel):
    topic: str = Field(
        default="වාරික ගණනය",
        description="Mathematics topic in Sinhala",
        examples=["වාරික ගණනය", "සරල පොලිය", "ලාභ හානි"]
    )
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.medium,
        description="Difficulty level"
    )
    num_questions: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of questions to generate (1-10)"
    )


class Question(BaseModel):
    question: str
    solution: str
    answer:  str


class QuestionResponse(BaseModel):
    success: bool
    topic: str
    difficulty: str
    questions: List[Question]
    count: int
    requested:  int
    generation_time_seconds: float
    model_used: str
    rag_context_used:  bool


class HealthResponse(BaseModel):
    status: str
    initialized: bool
    api_key_configured: bool
    model_name:  Optional[str]
    data_loaded: bool
    last_error: Optional[str]


class InitializeRequest(BaseModel):
    load_data: bool = Field(default=True, description="Whether to load RAG data")
    examples_path: str = Field(default="data/extracted_text/extracted_examples.json")
    exercises_path: str = Field(default="data/extracted_text/exteacted_exercises.json")
    paragraphs_path: str = Field(default="data/extracted_text/paragraphs_and_tables.json")
    guidelines_path: str = Field(default="data/extracted_text/guidelines.json")


# ==================== Helper Functions ====================

def get_rag_system() -> SinhalaRAGSystem: 
    """Get or initialize the RAG system"""
    global rag_system, system_status
    
    if rag_system is None: 
        if not GEMINI_API_KEY: 
            raise HTTPException(
                status_code=400,
                detail="GEMINI_API_KEY not configured.  Add it to your .env file."
            )
        
        try:
            print("\n🚀 Auto-initializing RAG system...")
            rag_system = SinhalaRAGSystem(api_key=GEMINI_API_KEY)
            system_status["initialized"] = True
            system_status["model_name"] = rag_system.model_name
            
            # Try to load data
            data_loaded = rag_system. load_all_data()
            system_status["data_loaded"] = data_loaded
            
        except Exception as e:
            system_status["last_error"] = str(e)
            raise HTTPException(status_code=500, detail=str(e))
    
    return rag_system


# ==================== API Endpoints ====================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Sinhala Math Question Generator API",
        "version": "2.0.0",
        "model":  "gemini-2.5-flash",
        "features": [
            "RAG with ChromaDB",
            "Context Retrieval",
            "Auto-retry",
            "Multilingual Embeddings"
        ],
        "status": "ready" if system_status["initialized"] else "waiting",
        "data_loaded": system_status. get("data_loaded", False),
        "default_questions": 5,
        "max_questions": 10,
        "docs":  "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API health status"""
    return HealthResponse(
        status="healthy" if GEMINI_API_KEY else "no_api_key",
        initialized=system_status["initialized"],
        api_key_configured=bool(GEMINI_API_KEY),
        model_name=system_status.get("model_name"),
        data_loaded=system_status.get("data_loaded", False),
        last_error=system_status.get("last_error")
    )


@app.post("/initialize", tags=["System"])
async def initialize_system(request: InitializeRequest = InitializeRequest()):
    """
    Initialize or reinitialize the RAG system
    
    - Loads the Gemini model
    - Optionally loads RAG data from JSON files
    """
    global rag_system, system_status
    
    if not GEMINI_API_KEY: 
        raise HTTPException(
            status_code=400,
            detail="GEMINI_API_KEY not configured."
        )
    
    try:
        print("\nInitializing RAG system...")
        rag_system = SinhalaRAGSystem(api_key=GEMINI_API_KEY)
        system_status["initialized"] = True
        system_status["model_name"] = rag_system.model_name
        system_status["last_error"] = None
        
        if request.load_data:
            data_loaded = rag_system. load_all_data(
                examples_path=request.examples_path,
                exercises_path=request.exercises_path,
                paragraphs_path=request.paragraphs_path,
                guidelines_path=request.guidelines_path
            )
            system_status["data_loaded"] = data_loaded
        
        return {
            "success": True,
            "message":  "RAG system initialized",
            "model":  rag_system.model_name,
            "data_loaded":  system_status.get("data_loaded", False)
        }
        
    except Exception as e:
        system_status["last_error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=QuestionResponse, tags=["Generate"])
async def generate_questions(request: QuestionRequest):
    """
    Generate Sinhala math questions using RAG
    
    - Uses context from loaded data files for better quality
    - Automatically retries until requested number is generated
    - Default:  5 questions, Maximum: 10 questions
    """
    rag = get_rag_system()
    
    start_time = time.time()
    
    try:
        questions, rag_used = rag.generate_questions(
            topic=request.topic,
            difficulty=request.difficulty. value,
            num_questions=request.num_questions
        )
        
        system_status["last_error"] = None
        
        return QuestionResponse(
            success=True,
            topic=request.topic,
            difficulty=request.difficulty.value,
            questions=[Question(**q) for q in questions],
            count=len(questions),
            requested=request.num_questions,
            generation_time_seconds=round(time.time() - start_time, 2),
            model_used=rag. model_name,
            rag_context_used=rag_used
        )
        
    except Exception as e:
        error_msg = str(e)
        system_status["last_error"] = error_msg
        
        if "quota" in error_msg. lower() or "rate" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Wait 2-3 minutes and try again."
            )
        else:
            raise HTTPException(status_code=500, detail=error_msg)


@app.post("/retrieve-context", tags=["RAG"])
async def retrieve_context(
    query: str,
    n_results: int = 3
):
    """
    Retrieve context from RAG system (for debugging/testing)
    
    - Shows what context would be used for a given query
    - Useful for understanding RAG retrieval
    """
    rag = get_rag_system()
    
    if not rag.data_loaded:
        raise HTTPException(
            status_code=400,
            detail="RAG data not loaded. Call /initialize with load_data=true first."
        )
    
    context = rag.retrieve_context(query, n_results)
    
    return {
        "query": query,
        "results": context,
        "total_items": sum(len(items) for items in context.values())
    }


@app.get("/topics", tags=["Reference"])
async def get_topics():
    """Get list of available mathematics topics"""
    return {
        "topics": [
            {"sinhala": "වාරික ගණනය", "english": "Installment Calculations"},
            {"sinhala":  "සරල පොලිය", "english": "Simple Interest"},
            {"sinhala": "සංයුක්ත පොලිය", "english": "Compound Interest"},
            {"sinhala": "ලාභ හානි", "english": "Profit and Loss"},
            {"sinhala": "ප්‍රතිශතය", "english": "Percentages"},
            {"sinhala": "අනුපාත", "english": "Ratios"},
        ],
        "default_questions": 5,
        "max_questions": 10
    }


@app. get("/data-status", tags=["System"])
async def get_data_status():
    """Get status of loaded RAG data"""
    global rag_system
    
    if rag_system is None:
        return {
            "initialized": False,
            "data_loaded": False,
            "collections": {}
        }
    
    return {
        "initialized": system_status["initialized"],
        "data_loaded": rag_system.data_loaded,
        "collections": rag_system.get_collection_stats(),
        "data_keys": list(rag_system. data. keys())
    }


# ==================== Startup Event ====================

@app.on_event("startup")
async def startup():
    """Initialize system on startup"""
    global rag_system, system_status
    
    print("\n" + "=" * 60)
    print("SINHALA MATH QUESTION GENERATOR API v2.0")
    print("With RAG (ChromaDB + Gemini 2.5 Flash)")
    print("=" * 60)
    print(f"API Key: {'Configured' if GEMINI_API_KEY else 'Not set'}")
    
    if GEMINI_API_KEY: 
        try:
            rag_system = SinhalaRAGSystem(api_key=GEMINI_API_KEY)
            system_status["initialized"] = True
            system_status["model_name"] = rag_system.model_name
            
            # Try to load data
            data_loaded = rag_system.load_all_data()
            system_status["data_loaded"] = data_loaded
            
            print(f"System ready!")
            print(f"Model: {rag_system.model_name}")
            print(f"RAG Data:  {'Loaded' if data_loaded else 'Not loaded'}")
            
        except Exception as e:
            print(f"Init error: {e}")
            system_status["last_error"] = str(e)
    
    print("=" * 60 + "\n")


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"\nStarting server at http://{host}:{port}")
    print(f"API Docs: http://{host}:{port}/docs\n")
    
    uvicorn.run("api:app", host=host, port=port, reload=True)