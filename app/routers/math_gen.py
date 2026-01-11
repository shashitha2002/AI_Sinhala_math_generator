import time
from fastapi import APIRouter, Depends, HTTPException
from app.models.math import QuestionRequest, QuestionResponse, Question
from app.dependencies import get_rag_system, get_current_user
from app.database import generated_questions_collection

# Note: We add dependencies=[Depends(get_current_user)] to protect ALL routes in this file
router = APIRouter(
    prefix="/math", 
    tags=["Math Generation"],
    dependencies=[Depends(get_current_user)] 
)

@router.post("/generate", response_model=QuestionResponse)
async def generate_questions(
    request: QuestionRequest,
    rag = Depends(get_rag_system),
    current_user: dict = Depends(get_current_user) # We can access user info here!
):
    print(f"User {current_user['email']} is generating questions...")
    
    start_time = time.time()
    try:
        # Assuming your RAG class has this method structure
        questions, rag_used = rag.generate_questions(
            topic=request.topic,
            difficulty=request.difficulty.value,
            num_questions=request.num_questions
        )
        
        result = {
            "user_email": current_user["email"],
            "topic": request.topic,
            "difficulty": request.difficulty.value,
            "questions": questions,
            "model_used": rag_used
        }

        # Save to DB
        await generated_questions_collection.insert_one(result)

        return QuestionResponse(
            success=True,
            topic=request.topic,
            questions=[Question(**q) for q in questions],
            generation_time_seconds=round(time.time() - start_time, 2),
            model_used=rag.model_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))