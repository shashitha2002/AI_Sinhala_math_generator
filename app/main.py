from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, math_gen
from app.dependencies import get_rag_system

app = FastAPI(title="Sinhala Math API v2", version="2.0.0")

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

# Startup Event to pre-load RAG data
@app.on_event("startup")
async def startup_event():
    try:
        rag = get_rag_system()
        rag.load_all_data() # Assuming this method exists in your class
        print("System Ready: RAG Data Loaded")
    except Exception as e:
        print(f"Warning: Could not auto-load RAG data: {e}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Secure Sinhala Math API"}