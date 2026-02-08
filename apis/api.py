"""
FastAPI Application for Sinhala Mathematics Question Generation
Uses the SinhalaRAGSystem and ModelPaperGenerator
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum
import os
import time
from dotenv import load_dotenv

# Import the RAG model and Model Paper Generator
from models.rag_model import SinhalaRAGSystem
from models.model_paper_generator import ModelPaperGenerator, ModelPaperConfig

load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Initialize FastAPI app
app = FastAPI(
    title="Sinhala Math Question Generator API",
    description="""
    ## O/L Mathematics Question Generation System
    
    ### Features:
    - **Lesson-wise Generation**: Generate questions for specific topics
    - **Model Paper Generation**: Generate complete exam papers (25 short + 5 structured + 10 essay)
    - **RAG System**: Uses past papers and examples for context
    
    ### Model Paper Structure:
    - 25 Short Answer Questions
    - 5 Structured Questions
    - 10 Essay Type Questions
    """,
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


# ==================== Global State ====================

rag_system: Optional[SinhalaRAGSystem] = None
model_paper_generator: Optional[ModelPaperGenerator] = None

system_status = {
    "initialized": False,
    "model_name": None,
    "last_error": None,
    "data_loaded": False,
    "past_papers_loaded": False
}

generation_status = {
    "is_generating": False,
    "current_task_id": None,
    "progress": {},
    "last_paper": None
}


# ==================== Pydantic Models - Lesson-wise ====================

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
    answer: str


class QuestionResponse(BaseModel):
    success: bool
    topic: str
    difficulty: str
    questions: List[Question]
    count: int
    requested: int
    generation_time_seconds: float
    model_used: str
    rag_context_used: bool


class HealthResponse(BaseModel):
    status: str
    initialized: bool
    api_key_configured: bool
    model_name: Optional[str]
    data_loaded: bool
    past_papers_loaded: bool
    last_error: Optional[str]


class InitializeRequest(BaseModel):
    load_data: bool = Field(default=True, description="Whether to load RAG data")
    load_past_papers: bool = Field(default=True, description="Whether to load past papers for model paper generation")
    examples_path: str = Field(default="data/extracted_text/extracted_examples.json")
    exercises_path: str = Field(default="data/extracted_text/exteacted_exercises.json")
    paragraphs_path: str = Field(default="data/extracted_text/paragraphs_and_tables.json")
    guidelines_path: str = Field(default="data/extracted_text/guidelines.json")
    past_papers_path: str = Field(default="data/extracted_text/model_paper_questions.json")


# ==================== Pydantic Models - Model Paper ====================

class AnswerStep(BaseModel):
    description: str = Field(..., description="What student should do")
    value: str = Field(..., description="The calculation or answer")


class SubQuestion(BaseModel):
    sub_question_label: str = Field(..., description="Label like (අ), (i)")
    sub_question: str = Field(..., description="Sub-question text")
    answer_steps: List[AnswerStep] = Field(default_factory=list)


class ShortAnswerQuestion(BaseModel):
    question_number: int
    question: str
    topics: List[str]
    answer_steps: List[AnswerStep]


class StructuredQuestion(BaseModel):
    question_number: int
    question: str
    topics: List[str]
    sub_questions: List[SubQuestion]


class EssayQuestion(BaseModel):
    question_number: int
    question: str
    topics: List[str]
    sub_questions: Optional[List[SubQuestion]] = None
    final_answer_steps: Optional[List[AnswerStep]] = None


class ModelPaperQuestions(BaseModel):
    short_answer: List[Dict]
    structured: List[Dict]
    essay_type: List[Dict]


class ModelPaperMetadata(BaseModel):
    topics_used: List[str]
    api_calls: int
    generation_time_seconds: float
    success_rate: Dict[str, Dict[str, int]]


class ModelPaperResponse(BaseModel):
    paper_id: str
    generated_at: str
    questions: ModelPaperQuestions
    metadata: ModelPaperMetadata


class GenerateModelPaperRequest(BaseModel):
    short_answer_count: int = Field(
        default=25,
        ge=1,
        le=25,
        description="Number of short answer questions (1-25)"
    )
    structured_count: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of structured questions (1-10)"
    )
    essay_count: int = Field(
        default=10,
        ge=1,
        le=10,
        description="Number of essay questions (1-10)"
    )
    api_delay: float = Field(
        default=4.0,
        ge=2.0,
        le=10.0,
        description="Delay between API calls in seconds"
    )


class GenerateTestPaperRequest(BaseModel):
    """Smaller request for testing"""
    short_answer_count: int = Field(default=3, ge=1, le=5)
    structured_count: int = Field(default=1, ge=1, le=2)
    essay_count: int = Field(default=1, ge=1, le=2)


class ProgressResponse(BaseModel):
    is_generating: bool
    task_id: Optional[str]
    current_type: Optional[str]
    questions_generated: int
    total_questions: int
    api_calls: int
    elapsed_seconds: float


# ==================== Helper Functions ====================

def get_rag_system() -> SinhalaRAGSystem:
    """Get or initialize the RAG system"""
    global rag_system, system_status
    
    if rag_system is None:
        if not GEMINI_API_KEY:
            raise HTTPException(
                status_code=400,
                detail="GEMINI_API_KEY not configured. Add it to your .env file."
            )
        
        try:
            print("\n🚀 Auto-initializing RAG system...")
            rag_system = SinhalaRAGSystem(api_key=GEMINI_API_KEY)
            system_status["initialized"] = True
            system_status["model_name"] = rag_system.model_name
            
            # Try to load data
            data_loaded = rag_system.load_all_data()
            system_status["data_loaded"] = data_loaded
            
        except Exception as e:
            system_status["last_error"] = str(e)
            raise HTTPException(status_code=500, detail=str(e))
    
    return rag_system


def get_model_paper_generator() -> ModelPaperGenerator:
    """Get or initialize the Model Paper Generator"""
    global model_paper_generator, system_status
    
    if model_paper_generator is None:
        if not GEMINI_API_KEY:
            raise HTTPException(
                status_code=400,
                detail="GEMINI_API_KEY not configured. Add it to your .env file."
            )
        
        try:
            print("\n🚀 Auto-initializing Model Paper Generator...")
            model_paper_generator = ModelPaperGenerator(api_key=GEMINI_API_KEY)
            system_status["initialized"] = True
            system_status["model_name"] = model_paper_generator.model_name
            
        except Exception as e:
            system_status["last_error"] = str(e)
            raise HTTPException(status_code=500, detail=str(e))
    
    return model_paper_generator


def update_progress(progress: Dict):
    """Callback for progress updates"""
    global generation_status
    generation_status["progress"] = progress


# ==================== General Endpoints ====================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Sinhala Math Question Generator API",
        "version": "2.0.0",
        "model": "gemini-2.5-flash",
        "features": [
            "Lesson-wise Question Generation",
            "Model Paper Generation",
            "RAG with ChromaDB",
            "Past Paper Reference",
            "Multilingual Embeddings"
        ],
        "status": "ready" if system_status["initialized"] else "waiting",
        "data_loaded": system_status.get("data_loaded", False),
        "past_papers_loaded": system_status.get("past_papers_loaded", False),
        "endpoints": {
            "lesson_wise": "/generate",
            "model_paper": "/model-paper/generate",
            "docs": "/docs"
        },
        "model_paper_structure": {
            "short_answer": 25,
            "structured": 5,
            "essay_type": 10
        }
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
        past_papers_loaded=system_status.get("past_papers_loaded", False),
        last_error=system_status.get("last_error")
    )


@app.post("/initialize", tags=["System"])
async def initialize_system(request: InitializeRequest = InitializeRequest()):
    """
    Initialize or reinitialize the RAG system and Model Paper Generator
    
    - Loads the Gemini model
    - Optionally loads RAG data from JSON files
    - Optionally loads past papers for model paper generation
    """
    global rag_system, model_paper_generator, system_status
    
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="GEMINI_API_KEY not configured."
        )
    
    try:
        print("\n🚀 Initializing systems...")
        
        # Initialize RAG system
        rag_system = SinhalaRAGSystem(api_key=GEMINI_API_KEY)
        system_status["initialized"] = True
        system_status["model_name"] = rag_system.model_name
        system_status["last_error"] = None
        
        # Load RAG data
        if request.load_data:
            data_loaded = rag_system.load_all_data(
                examples_path=request.examples_path,
                exercises_path=request.exercises_path,
                paragraphs_path=request.paragraphs_path,
                guidelines_path=request.guidelines_path
            )
            system_status["data_loaded"] = data_loaded
        
        # Initialize Model Paper Generator
        model_paper_generator = ModelPaperGenerator(api_key=GEMINI_API_KEY)
        
        # Load past papers
        if request.load_past_papers:
            past_papers_loaded = model_paper_generator.load_past_paper_questions(
                request.past_papers_path
            )
            system_status["past_papers_loaded"] = past_papers_loaded
        
        return {
            "success": True,
            "message": "Systems initialized",
            "model": rag_system.model_name,
            "data_loaded": system_status.get("data_loaded", False),
            "past_papers_loaded": system_status.get("past_papers_loaded", False),
            "available_topics": model_paper_generator.available_topics if model_paper_generator else []
        }
        
    except Exception as e:
        system_status["last_error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data-status", tags=["System"])
async def get_data_status():
    """Get status of loaded data"""
    global rag_system, model_paper_generator
    
    result = {
        "initialized": system_status["initialized"],
        "rag_data_loaded": False,
        "past_papers_loaded": False,
        "collections": {},
        "available_topics": []
    }
    
    if rag_system is not None:
        result["rag_data_loaded"] = rag_system.data_loaded
        result["collections"] = rag_system.get_collection_stats()
    
    if model_paper_generator is not None:
        result["past_papers_loaded"] = model_paper_generator.past_papers_loaded
        result["available_topics"] = model_paper_generator.available_topics
        if model_paper_generator.past_papers_loaded:
            stats = model_paper_generator.get_statistics()
            result["past_paper_stats"] = stats
    
    return result


# ==================== Lesson-wise Generation Endpoints ====================

@app.post("/generate", response_model=QuestionResponse, tags=["Lesson-wise Generation"])
async def generate_questions(request: QuestionRequest):
    """
    Generate Sinhala math questions for a specific topic
    
    - Uses context from loaded data files for better quality
    - Automatically retries until requested number is generated
    - Default: 5 questions, Maximum: 10 questions
    """
    rag = get_rag_system()
    
    start_time = time.time()
    
    try:
        questions, rag_used = rag.generate_questions(
            topic=request.topic,
            difficulty=request.difficulty.value,
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
            model_used=rag.model_name,
            rag_context_used=rag_used
        )
        
    except Exception as e:
        error_msg = str(e)
        system_status["last_error"] = error_msg
        
        if "quota" in error_msg.lower() or "rate" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Wait 2-3 minutes and try again."
            )
        else:
            raise HTTPException(status_code=500, detail=error_msg)


@app.post("/retrieve-context", tags=["Lesson-wise Generation"])
async def retrieve_context(query: str, n_results: int = 3):
    """
    Retrieve context from RAG system (for debugging/testing)
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
    global model_paper_generator
    
    # Static topics for lesson-wise
    lesson_topics = [
        {"sinhala": "පොළිය", "english": "Interest"},
        {"sinhala": "සමීකරණ", "english": "Equations"},
        {"sinhala": "කොටස් වෙළෙඳපොළ", "english": "Stock Market"},
        {"sinhala": "ලඝුගණක", "english": "Logarithms"},
        {"sinhala": "ශ්‍රීඝ්‍රතාවය", "english": "Speed"},
        {"sinhala": "සමාන්තර ශ්‍රේණි", "english": "Arithmetic Progression"},
    ]
    
    # Topics from past papers
    past_paper_topics = []
    if model_paper_generator and model_paper_generator.past_papers_loaded:
        past_paper_topics = model_paper_generator.available_topics
    
    return {
        "lesson_topics": lesson_topics,
        "past_paper_topics": past_paper_topics,
        "default_questions": 5,
        "max_questions": 10
    }


# ==================== Model Paper Generation Endpoints ====================

@app.get("/model-paper/status", tags=["Model Paper Generation"])
async def get_model_paper_status():
    """Get current model paper generator status"""
    global model_paper_generator, generation_status
    
    return {
        "initialized": model_paper_generator is not None,
        "past_papers_loaded": model_paper_generator.past_papers_loaded if model_paper_generator else False,
        "is_generating": generation_status["is_generating"],
        "available_topics": len(model_paper_generator.available_topics) if model_paper_generator else 0
    }


@app.get("/model-paper/topics", tags=["Model Paper Generation"])
async def get_model_paper_topics():
    """Get list of available topics from loaded past papers"""
    generator = get_model_paper_generator()
    
    if not generator.past_papers_loaded:
        raise HTTPException(
            status_code=400,
            detail="Past papers not loaded. Call /initialize with load_past_papers=true first."
        )
    
    stats = generator.get_statistics()
    
    return {
        "available_topics": stats["available_topics"],
        "total_topics": len(stats["available_topics"]),
        "questions_by_topic": stats["questions_by_topic"],
        "questions_by_type": stats["questions_by_type"]
    }


@app.post("/model-paper/generate", response_model=ModelPaperResponse, tags=["Model Paper Generation"])
async def generate_model_paper(request: GenerateModelPaperRequest = GenerateModelPaperRequest()):
    """
    Generate a complete O/L Mathematics model paper
    
    ⚠️ This is a long-running operation (2-5 minutes)
    
    - Generates 25 short answer + 5 structured + 10 essay questions by default
    - Uses multiple API calls due to token limits
    - Returns complete paper with questions and answers
    
    ### Output Format:
    Each question has:
    - `question`: The question text
    - `topics`: List of topics covered
    - `answer_steps`: List of {description, value} pairs
    
    For web display:
    ```
    description _____________ (input field for student)
    ```
    """
    global generation_status
    
    generator = get_model_paper_generator()
    
    if not generator.past_papers_loaded:
        raise HTTPException(
            status_code=400,
            detail="Past papers not loaded. Call /initialize with load_past_papers=true first."
        )
    
    if generation_status["is_generating"]:
        raise HTTPException(
            status_code=409,
            detail="Generation already in progress. Check /model-paper/progress endpoint."
        )
    
    # Set status
    generation_status["is_generating"] = True
    generation_status["current_task_id"] = f"gen_{int(time.time())}"
    generation_status["progress"] = {"started_at": time.time()}
    
    try:
        # Configure
        config = ModelPaperConfig(
            short_answer_count=request.short_answer_count,
            structured_count=request.structured_count,
            essay_count=request.essay_count,
            api_delay=request.api_delay
        )
        
        # Generate
        model_paper = generator.generate_model_paper(
            config=config,
            progress_callback=update_progress
        )
        
        generation_status["last_paper"] = model_paper
        
        return ModelPaperResponse(
            paper_id=model_paper["paper_id"],
            generated_at=model_paper["generated_at"],
            questions=ModelPaperQuestions(
                short_answer=model_paper["questions"]["short_answer"],
                structured=model_paper["questions"]["structured"],
                essay_type=model_paper["questions"]["essay_type"]
            ),
            metadata=ModelPaperMetadata(
                topics_used=model_paper["metadata"]["topics_used"],
                api_calls=model_paper["metadata"]["api_calls"],
                generation_time_seconds=model_paper["metadata"]["generation_time_seconds"],
                success_rate=model_paper["metadata"]["success_rate"]
            )
        )
        
    except Exception as e:
        error_msg = str(e)
        system_status["last_error"] = error_msg
        
        if "quota" in error_msg.lower() or "rate" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Wait 2-3 minutes and try again."
            )
        else:
            raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        generation_status["is_generating"] = False


@app.post("/model-paper/generate-test", tags=["Model Paper Generation"])
async def generate_test_paper(request: GenerateTestPaperRequest = GenerateTestPaperRequest()):
    """
    Generate a small test paper (for testing purposes)
    
    - Default: 3 short answer + 1 structured + 1 essay
    - Faster than full paper generation (~1-2 minutes)
    """
    full_request = GenerateModelPaperRequest(
        short_answer_count=request.short_answer_count,
        structured_count=request.structured_count,
        essay_count=request.essay_count,
        api_delay=4.0
    )
    
    return await generate_model_paper(full_request)


@app.get("/model-paper/progress", response_model=ProgressResponse, tags=["Model Paper Generation"])
async def get_generation_progress():
    """Get current generation progress"""
    global generation_status
    
    progress = generation_status.get("progress", {})
    started_at = progress.get("started_at", 0)
    elapsed = time.time() - started_at if started_at else 0
    
    return ProgressResponse(
        is_generating=generation_status["is_generating"],
        task_id=generation_status.get("current_task_id"),
        current_type=progress.get("current_type"),
        questions_generated=progress.get("generated", 0),
        total_questions=progress.get("total", 0),
        api_calls=progress.get("api_calls", 0),
        elapsed_seconds=round(elapsed, 2)
    )


@app.get("/model-paper/last", tags=["Model Paper Generation"])
async def get_last_generated_paper():
    """Get the last generated model paper"""
    global generation_status
    
    if generation_status["last_paper"] is None:
        raise HTTPException(
            status_code=404,
            detail="No paper generated yet"
        )
    
    return generation_status["last_paper"]


@app.get("/model-paper/sample-output", tags=["Model Paper Generation"])
async def get_sample_output():
    """
    Get a sample of the output format (no authentication required)
    
    Useful for understanding the API response structure for frontend development
    """
    return {
        "paper_id": "MP_20260206_143022",
        "generated_at": "2026-02-06 14:30:22",
        "questions": {
            "short_answer": [
                {
                    "question_number": 1,
                    "question": "සුළු කරන්න: (3/4x) + (2/3x) - (1/6x)",
                    "topics": ["වීජීය භාග"],
                    "answer_steps": [
                        {"description": "පොදු හරය සොයන්න", "value": "12x"},
                        {"description": "භාග සමාන කරන්න", "value": "9/12x + 8/12x - 2/12x"},
                        {"description": "අවසාන පිළිතුර", "value": "15/12x = 5/4x"}
                    ]
                }
            ],
            "structured": [
                {
                    "question_number": 1,
                    "question": "රවී රුපියල් 50000 ක් බැංකුවක 8% වාර්ෂික පොලී අනුපාතයකට තැන්පත් කරයි.",
                    "topics": ["පොලිය"],
                    "sub_questions": [
                        {
                            "sub_question_label": "(අ)",
                            "sub_question": "පළමු වසර අවසානයේ ලැබෙන පොලිය සොයන්න",
                            "answer_steps": [
                                {"description": "සූත්‍රය යොදන්න", "value": "P×R×T/100 = 50000×8×1/100"},
                                {"description": "අවසාන පිළිතුර", "value": "රු. 4000"}
                            ]
                        },
                        {
                            "sub_question_label": "(ආ)",
                            "sub_question": "දෙවන වසර අවසානයේ මුළු මුදල සොයන්න",
                            "answer_steps": [
                                {"description": "පළමු වසරේ මුදල", "value": "50000 + 4000 = 54000"},
                                {"description": "දෙවන වසරේ පොලිය", "value": "54000 × 8/100 = 4320"},
                                {"description": "මුළු මුදල", "value": "රු. 58320"}
                            ]
                        }
                    ]
                }
            ],
            "essay_type": [
                {
                    "question_number": 1,
                    "question": "පැත්තක දිග a වූ ඝනකය�� පරිමාව සහ පෘෂ්ඨ වර්ගඵලය ගණනය කිරීමට ලඝුගණක භාවිතා කරන්න.",
                    "topics": ["ඝන වස්තුවල පරිමාව", "ලඝුගණක"],
                    "sub_questions": [
                        {
                            "sub_question_label": "(i)",
                            "sub_question": "ඝනකයේ පරිමාව a³ බව පෙන්වන්න",
                            "answer_steps": [
                                {"description": "පරිමා සූත්‍රය", "value": "V = a × a × a = a³"}
                            ]
                        },
                        {
                            "sub_question_label": "(ii)",
                            "sub_question": "a = 2.5 cm නම් lg භාවිතයෙන් පරිමාව සොයන්න",
                            "answer_steps": [
                                {"description": "lg V = 3 × lg a", "value": "3 × lg 2.5"},
                                {"description": "lg 2.5 සොයන්න", "value": "0.3979"},
                                {"description": "3 × 0.3979", "value": "1.1937"},
                                {"description": "antilog සොයන්න", "value": "15.625 cm³"}
                            ]
                        }
                    ],
                    "final_answer_steps": None
                }
            ]
        },
        "metadata": {
            "topics_used": ["වීජීය භාග", "පොලිය", "ඝන වස්තුවල පරිමාව", "ලඝුගණක"],
            "api_calls": 15,
            "generation_time_seconds": 180.5,
            "success_rate": {
                "short_answer": {"requested": 25, "generated": 25},
                "structured": {"requested": 5, "generated": 5},
                "essay_type": {"requested": 10, "generated": 10}
            }
        },
        "_web_display_hint": {
            "format": "description _____________ (input field)",
            "example": "පොදු හරය සොයන්න _____________ [Student enters: 12x]"
        }
    }


# ==================== Background Generation (Optional) ====================

async def generate_paper_background(
    config: ModelPaperConfig,
    generator: ModelPaperGenerator,
    task_id: str
):
    """Background task for paper generation"""
    global generation_status
    
    try:
        model_paper = generator.generate_model_paper(
            config=config,
            progress_callback=update_progress
        )
        generation_status["last_paper"] = model_paper
        
    except Exception as e:
        generation_status["progress"]["error"] = str(e)
    
    finally:
        generation_status["is_generating"] = False


@app.post("/model-paper/generate-async", tags=["Model Paper Generation"])
async def generate_model_paper_async(
    background_tasks: BackgroundTasks,
    request: GenerateModelPaperRequest = GenerateModelPaperRequest()
):
    """
    Start model paper generation in background
    
    - Returns immediately with task_id
    - Check /model-paper/progress for status
    - Get result from /model-paper/last when complete
    """
    global generation_status
    
    generator = get_model_paper_generator()
    
    if not generator.past_papers_loaded:
        raise HTTPException(
            status_code=400,
            detail="Past papers not loaded. Call /initialize with load_past_papers=true first."
        )
    
    if generation_status["is_generating"]:
        raise HTTPException(
            status_code=409,
            detail="Generation already in progress"
        )
    
    task_id = f"gen_{int(time.time())}"
    generation_status["is_generating"] = True
    generation_status["current_task_id"] = task_id
    generation_status["progress"] = {"started_at": time.time()}
    
    config = ModelPaperConfig(
        short_answer_count=request.short_answer_count,
        structured_count=request.structured_count,
        essay_count=request.essay_count,
        api_delay=request.api_delay
    )
    
    background_tasks.add_task(
        generate_paper_background,
        config,
        generator,
        task_id
    )
    
    return {
        "success": True,
        "task_id": task_id,
        "message": "Generation started. Check /model-paper/progress for status.",
        "estimated_time_minutes": (request.short_answer_count // 5 + request.structured_count + request.essay_count // 2) * 0.5
    }


# ==================== Startup Event ====================

@app.on_event("startup")
async def startup():
    """Initialize system on startup"""
    global rag_system, model_paper_generator, system_status
    
    print("\n" + "=" * 60)
    print("🎓 SINHALA MATH QUESTION GENERATOR API v2.0")
    print("   With RAG + Model Paper Generation")
    print("=" * 60)
    print(f"API Key: {'✅ Configured' if GEMINI_API_KEY else '❌ Not set'}")
    
    if GEMINI_API_KEY:
        try:
            # Initialize RAG system
            rag_system = SinhalaRAGSystem(api_key=GEMINI_API_KEY)
            system_status["initialized"] = True
            system_status["model_name"] = rag_system.model_name
            
            # Load RAG data
            data_loaded = rag_system.load_all_data()
            system_status["data_loaded"] = data_loaded
            
            # Initialize Model Paper Generator
            model_paper_generator = ModelPaperGenerator(api_key=GEMINI_API_KEY)
            
            # Load past papers
            past_papers_path = "data/extracted_text/model_paper_questions.json"
            if os.path.exists(past_papers_path):
                past_papers_loaded = model_paper_generator.load_past_paper_questions(past_papers_path)
                system_status["past_papers_loaded"] = past_papers_loaded
            else:
                print(f"⚠️ Past papers file not found: {past_papers_path}")
            
            print(f"\n✅ System ready!")
            print(f"   Model: {rag_system.model_name}")
            print(f"   RAG Data: {'✅ Loaded' if data_loaded else '❌ Not loaded'}")
            print(f"   Past Papers: {'✅ Loaded' if system_status.get('past_papers_loaded') else '❌ Not loaded'}")
            if model_paper_generator.available_topics:
                print(f"   Topics: {len(model_paper_generator.available_topics)} available")
            
        except Exception as e:
            print(f"❌ Init error: {e}")
            system_status["last_error"] = str(e)
    
    print("=" * 60)
    print("📚 Endpoints:")
    print("   /docs          - API Documentation")
    print("   /generate      - Lesson-wise questions")
    print("   /model-paper/* - Model paper generation")
    print("=" * 60 + "\n")


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"\n🚀 Starting server at http://{host}:{port}")
    print(f"📖 API Docs: http://{host}:{port}/docs\n")
    
    uvicorn.run("api:app", host=host, port=port, reload=True)