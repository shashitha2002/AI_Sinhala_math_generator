from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, math_gen, forum, quiz, gemini, progress, badges, recommendations, games, math_quiz
from app.dependencies import get_rag_system

app = FastAPI(title="Sinhala Math API v2", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect Routers
app.include_router(auth.router, prefix="/api")
app.include_router(math_gen.router, prefix="/api")
app.include_router(forum.router, prefix="/api")
app.include_router(quiz.router, prefix="/api")
app.include_router(gemini.router, prefix="/api")
app.include_router(progress.router, prefix="/api")
app.include_router(badges.router, prefix="/api")
app.include_router(recommendations.router, prefix="/api")
app.include_router(games.router, prefix="/api")
app.include_router(math_quiz.router, prefix="/api")

# Startup Event to pre-load RAG data
@app.on_event("startup")
async def startup_event():
    print("Initializing RAG System...")
    try:
        rag = get_rag_system()
        # Load RAG data from textbook files
        success = rag.load_all_data()
        if success:
            print("✅ System Ready: RAG Data Loaded")
        else:
            print("⚠️ Warning: RAG data files not found, using fallback generation")
    except Exception as e:
        print(f"⚠️ Warning: Could not auto-load RAG data: {e}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Secure Sinhala Math API"}