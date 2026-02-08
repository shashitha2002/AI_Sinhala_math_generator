from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, math_gen, model_paper 
from app.dependencies import get_rag_system, get_model_paper_generator  

app = FastAPI(
    title="Sinhala Math API v2", 
    version="2.0.0",
    description="""
    ## O/L Mathematics Question Generation System
    
    ### Features:
    - **Authentication**: User registration and login
    - **Lesson-wise Generation**: Generate questions for specific topics
    - **Model Paper Generation**: Generate complete exam papers
    """
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect Routers
app.include_router(auth.router)
app.include_router(math_gen.router)
app.include_router(model_paper.router)  # <-- Add this line

# Startup Event to pre-load RAG data
@app.on_event("startup")
async def startup_event():
    try:
        # Load RAG data
        rag = get_rag_system()
        rag.load_all_data()
        print("✅ RAG Data Loaded")
        
        # Load Model Paper Generator data
        generator = get_model_paper_generator()
        generator.load_past_paper_questions("data/extracted_text/model_paper_questions.json")
        print("✅ Past Papers Loaded")
        print(f"   Available Topics: {len(generator.available_topics)}")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not auto-load data: {e}")

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Secure Sinhala Math API",
        "version": "2.0.0",
        "endpoints": {
            "auth": "/auth/*",
            "lesson_wise": "/math/*",
            "model_paper": "/model-paper/*",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}