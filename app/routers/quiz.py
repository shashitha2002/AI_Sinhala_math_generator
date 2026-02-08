from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from app.database import quizzes_collection, users_collection, syllabus_topics_collection
from app.models.quiz import Quiz, QuizQuestion, QuizAnswer, QuizScore, QuizStatus
from app.models.utils import PyObjectId
from app.dependencies import get_current_user
from datetime import datetime
from bson import ObjectId
import random

router = APIRouter(prefix="/quiz", tags=["Quiz"])

@router.get("/generate")
async def generate_quiz(
    topic: Optional[str] = None,
    difficulty: str = "medium",
    type: str = "adaptive",
    current_user: dict = Depends(get_current_user)
):
    user_id = str(current_user["_id"])
    
    # In a real scenario, we might call the RAG system or an ML service here.
    # For now, we'll try to find questions in the syllabus_topics_collection or return dummy questions.
    
    questions = []
    # Dummy questions for demonstration
    topics = ["Algebra", "Geometry", "Trigonometry"]
    selected_topic = topic or random.choice(topics)
    
    for i in range(5):
        questions.append({
            "questionId": f"q_{i}",
            "question": f"Sample {selected_topic} question {i+1} in Sinhala?",
            "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
            "correctAnswer": random.randint(0, 3),
            "explanation": "This is a sample explanation.",
            "topic": selected_topic,
            "difficulty": difficulty,
            "marks": 1
        })
    
    quiz = {
        "userId": ObjectId(user_id),
        "type": type,
        "questions": questions,
        "answers": [],
        "status": "in-progress",
        "timeStarted": datetime.utcnow(),
        "createdAt": datetime.utcnow()
    }
    
    result = await quizzes_collection.insert_one(quiz)
    
    return {
        "success": True,
        "quiz": {
            "id": str(result.inserted_id),
            "questions": questions,
            "type": type
        }
    }

@router.post("/submit")
async def submit_quiz(
    quiz_id: str, 
    answers: List[dict],
    current_user: dict = Depends(get_current_user)
):
    user_id = str(current_user["_id"])
    
    if not ObjectId.is_valid(quiz_id):
        raise HTTPException(status_code=400, detail="Invalid Quiz ID")
    
    quiz = await quizzes_collection.find_one({"_id": ObjectId(quiz_id), "userId": ObjectId(user_id)})
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    correct_count = 0
    results = []
    
    for i, answer in enumerate(answers):
        question = quiz["questions"][i]
        is_correct = answer["selectedAnswer"] == question["correctAnswer"]
        if is_correct:
            correct_count += 1
            
        results.append({
            "questionId": question["questionId"],
            "selectedAnswer": answer["selectedAnswer"],
            "isCorrect": is_correct,
            "timeSpent": answer.get("timeSpent", 0),
            "timestamp": datetime.utcnow()
        })
        
    total_questions = len(quiz["questions"])
    percentage = (correct_count / total_questions) * 100
    
    score = {
        "total": total_questions,
        "obtained": correct_count,
        "percentage": percentage
    }
    
    await quizzes_collection.update_one(
        {"_id": ObjectId(quiz_id)},
        {
            "$set": {
                "answers": results,
                "score": score,
                "status": "completed",
                "timeCompleted": datetime.utcnow()
            }
        }
    )
    
    # Update user performance summary
    await users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {
            "$inc": {"performance.totalQuizzes": 1},
            "$set": {"performance.lastQuizDate": datetime.utcnow()} # Simplified
        }
    )
    
    return {
        "success": True,
        "score": score,
        "results": results
    }

@router.get("/history")
async def get_history(current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    
    cursor = quizzes_collection.find({"userId": ObjectId(user_id)}).sort("createdAt", -1).limit(20)
    quizzes = await cursor.to_list(length=20)
    
    for q in quizzes:
        q["_id"] = str(q["_id"])
        q["userId"] = str(q["userId"])
        
    return {"success": True, "quizzes": quizzes}

@router.post("/evaluate/quiz")
async def evaluate_quiz(
    questions: List[dict],
    current_user: dict = Depends(get_current_user)
):
    # This matches the Node backend logic but in Python
    # It would ideally use the RAG or evaluation system
    results = []
    for q in questions:
        # Simple evaluation for now or call AI
        is_correct = q.get("userFinalAnswer") == q.get("correctAnswer")
        results.append({
            "isCorrect": is_correct,
            "marksObtained": 1 if is_correct else 0,
            "feedback": "Smarter evaluation coming soon..."
        })
    
    return {"success": True, "results": results}

@router.get("/topics")
async def get_topics():
    topics = await syllabus_topics_collection.find().to_list(length=100)
    for t in topics:
        t["_id"] = str(t["_id"])
    return {"success": True, "topics": topics}

@router.get("/model-paper")
async def get_model_paper(current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    
    # Generate a larger quiz for model paper
    # Real implementation would mix topics
    questions = []
    topics = ["Algebra", "Geometry", "Trigonometry", "Statistics", "Mensuration"]
    
    for i in range(10): # Model paper has 10 questions for now
        topic = topics[i % len(topics)]
        questions.append({
            "questionId": f"mp_q_{i}",
            "question": f"Model Paper Question {i+1} on {topic}?",
            "options": ["A", "B", "C", "D"],
            "correctAnswer": 0,
            "explanation": "Explanation here",
            "topic": topic,
            "difficulty": "hard",
            "marks": 2
        })
        
    quiz = {
        "userId": ObjectId(user_id),
        "type": "model-paper",
        "questions": questions,
        "answers": [],
        "status": "in-progress",
        "timeStarted": datetime.utcnow(),
        "createdAt": datetime.utcnow()
    }
    
    result = await quizzes_collection.insert_one(quiz)
    
    return {
        "success": True,
        "quiz": {
            "id": str(result.inserted_id),
            "questions": questions,
            "type": "model-paper"
        }
    }

@router.get("/{quiz_id}")
async def get_quiz_by_id(quiz_id: str, current_user: dict = Depends(get_current_user)):
    if not ObjectId.is_valid(quiz_id):
        raise HTTPException(status_code=400, detail="Invalid Quiz ID")
        
    quiz = await quizzes_collection.find_one({
        "_id": ObjectId(quiz_id),
        "userId": ObjectId(current_user["_id"])
    })
    
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
        
    quiz["_id"] = str(quiz["_id"])
    quiz["userId"] = str(quiz["userId"])
    
    return {"success": True, "quiz": quiz}
