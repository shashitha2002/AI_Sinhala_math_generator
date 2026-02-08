from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Any, Dict
from app.database import math_quizzes_collection, users_collection
from app.dependencies import get_current_user
from datetime import datetime
from bson import ObjectId
from pydantic import BaseModel, Field, validator

router = APIRouter(prefix="/math-quiz", tags=["MathQuiz"])

# Models
class Step(BaseModel):
    header: str
    description: str
    calculation: str
    answer: str

class MathQuizQuestion(BaseModel):
    question: str
    answer: str
    solution: str
    steps: List[Step] = []
    rag_context_used: Optional[bool] = False

class MathQuizSaveRequest(BaseModel):
    topic: str
    topicEnglish: Optional[str] = None
    difficulty: str
    questions: List[MathQuizQuestion]
    rag_context_used: Optional[bool] = False

class UserStepAnswer(BaseModel):
    stepAnswers: List[str]
    finalAnswer: str
    timeSpent: float
    marksObtained: Optional[float] = 0
    isCorrect: Optional[bool] = False
    feedback: Optional[str] = None
    explanation: Optional[str] = None

class SubmitQuizRequest(BaseModel):
    userAnswers: List[UserStepAnswer]
    evaluationResults: List[Any] = []

# Routes

@router.post("/save")
async def save_quiz(
    quiz_data: MathQuizSaveRequest,
    current_user: dict = Depends(get_current_user)
):
    user_id = str(current_user["_id"])
    
    # Check if there is already an in-progress quiz for this topic/difficulty? 
    # Maybe not strict, but good practice. For now, just create new.
    
    new_quiz = {
        "userId": ObjectId(user_id),
        "topic": quiz_data.topic,
        "topicEnglish": quiz_data.topicEnglish,
        "difficulty": quiz_data.difficulty,
        "questions": [q.dict() for q in quiz_data.questions],
        "rag_context_used": quiz_data.rag_context_used,
        "status": "not-started",
        "createdAt": datetime.utcnow(),
        "userAnswers": [],
        "score": None
    }
    
    result = await math_quizzes_collection.insert_one(new_quiz)
    
    return {
        "success": True,
        "quizId": str(result.inserted_id),
        "message": "Quiz saved successfully"
    }

@router.get("/current")
async def get_current_quiz(
    topic: str,
    difficulty: str,
    current_user: dict = Depends(get_current_user)
):
    user_id = str(current_user["_id"])
    
    # Find latest in-progress or not-started quiz
    quiz = await math_quizzes_collection.find_one({
        "userId": ObjectId(user_id),
        "topic": topic,
        "difficulty": difficulty,
        "status": {"$in": ["not-started", "in-progress"]}
    }, sort=[("createdAt", -1)])
    
    if not quiz:
        raise HTTPException(status_code=404, detail="No active quiz found")
        
    quiz["id"] = str(quiz["_id"])
    del quiz["_id"]
    quiz["userId"] = str(quiz["userId"])
    
    return quiz

@router.get("/history/all")
async def get_quiz_history(
    status: Optional[str] = None,
    topic: Optional[str] = None,
    limit: int = 20,
    current_user: dict = Depends(get_current_user)
):
    user_id = str(current_user["_id"])
    query = {"userId": ObjectId(user_id)}
    
    if status:
        query["status"] = status
    if topic:
        query["topic"] = topic
        
    cursor = math_quizzes_collection.find(query).sort("createdAt", -1).limit(limit)
    quizzes = await cursor.to_list(length=limit)
    
    results = []
    for q in quizzes:
        results.append({
            "id": str(q["_id"]),
            "topic": q.get("topic"),
            "topicEnglish": q.get("topicEnglish"),
            "difficulty": q.get("difficulty"),
            "status": q.get("status"),
            "score": q.get("score"),
            "questionsCount": len(q.get("questions", [])),
            "createdAt": q.get("createdAt"),
            "timeCompleted": q.get("timeCompleted"),
            "totalTimeSpent": q.get("totalTimeSpent")
        })
        
    return {"success": True, "history": results}

@router.get("/stats/summary")
async def get_quiz_stats(current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    
    pipeline = [
        {"$match": {"userId": ObjectId(user_id), "status": "completed"}},
        {"$group": {
            "_id": None,
            "totalQuizzes": {"$sum": 1},
            "totalQuestions": {"$sum": {"$size": "$questions"}},
            "totalTime": {"$sum": "$totalTimeSpent"},
            "avgPercentage": {"$avg": "$score.percentage"}
        }}
    ]
    
    stats = await math_quizzes_collection.aggregate(pipeline).to_list(length=1)
    
    # Calculate total correct separately or refine pipeline
    # For now keep it simple
    total_correct = 0
    # We might need a separate query for total correct answers if structure is complex
    
    if stats:
        stat = stats[0]
        return {
            "totalQuizzes": stat.get("totalQuizzes", 0),
            "averageScore": stat.get("avgPercentage", 0),
            "totalTimeSpent": stat.get("totalTime", 0),
            "totalQuestions": stat.get("totalQuestions", 0),
            "totalCorrect": 0 # TODO: Calculate this
        }
    
    return {
        "totalQuizzes": 0,
        "averageScore": 0,
        "totalTimeSpent": 0,
        "totalQuestions": 0,
        "totalCorrect": 0
    }

@router.get("/{quiz_id}")
async def get_quiz_by_id(quiz_id: str, current_user: dict = Depends(get_current_user)):
    if not ObjectId.is_valid(quiz_id):
        raise HTTPException(status_code=400, detail="Invalid Quiz ID")
        
    quiz = await math_quizzes_collection.find_one({
        "_id": ObjectId(quiz_id),
        "userId": ObjectId(current_user["_id"])
    })
    
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
        
    quiz["id"] = str(quiz["_id"])
    del quiz["_id"]
    quiz["userId"] = str(quiz["userId"])
    
    return quiz

@router.put("/{quiz_id}/start")
async def start_quiz(quiz_id: str, current_user: dict = Depends(get_current_user)):
    if not ObjectId.is_valid(quiz_id):
        raise HTTPException(status_code=400, detail="Invalid Quiz ID")
        
    result = await math_quizzes_collection.update_one(
        {"_id": ObjectId(quiz_id), "userId": ObjectId(current_user["_id"])},
        {"$set": {
            "status": "in-progress",
            "timeStarted": datetime.utcnow()
        }}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Quiz not found")
        
    return {"success": True}

@router.put("/{quiz_id}/submit")
async def submit_quiz(
    quiz_id: str,
    submit_data: SubmitQuizRequest,
    current_user: dict = Depends(get_current_user)
):
    if not ObjectId.is_valid(quiz_id):
        raise HTTPException(status_code=400, detail="Invalid Quiz ID")
        
    user_id = str(current_user["_id"])
    
    quiz = await math_quizzes_collection.find_one({
        "_id": ObjectId(quiz_id), 
        "userId": ObjectId(user_id)
    })
    
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
        
    # Calculate Score
    user_answers = submit_data.userAnswers
    questions = quiz.get("questions", [])
    
    correct_count = 0
    total_marks = 0
    obtained_marks = 0
    
    processed_answers = []
    
    for i, ans in enumerate(user_answers):
        # In a real app, verify correctness again? 
        # For now trust the client's isCorrect or re-evaluate. 
        # The client sends 'isCorrect' in UserAnswer based on self-check or previous verify step.
        
        is_correct = ans.isCorrect
        if is_correct:
            correct_count += 1
            
        marks = ans.marksObtained or (10 if is_correct else 0) # Default marks
        
        obtained_marks += marks
        total_marks += 10 # Default 10 marks per question
        
        processed_answers.append(ans.dict())
        
    total_questions = len(questions)
    percentage = (correct_count / total_questions * 100) if total_questions > 0 else 0
    
    score = {
        "totalQuestions": total_questions,
        "correctAnswers": correct_count,
        "totalMarks": total_marks,
        "obtainedMarks": obtained_marks,
        "percentage": percentage
    }
    
    # Update DB
    await math_quizzes_collection.update_one(
        {"_id": ObjectId(quiz_id)},
        {"$set": {
            "status": "completed",
            "timeCompleted": datetime.utcnow(),
            "userAnswers": processed_answers,
            "score": score,
            # Calculate total time spent from answers
            "totalTimeSpent": sum(ans.timeSpent for ans in user_answers)
        }}
    )
    
    return {
        "success": True,
        "score": score
    }

@router.delete("/{quiz_id}")
async def delete_quiz(quiz_id: str, current_user: dict = Depends(get_current_user)):
    if not ObjectId.is_valid(quiz_id):
        raise HTTPException(status_code=400, detail="Invalid Quiz ID")
        
    result = await math_quizzes_collection.delete_one({
        "_id": ObjectId(quiz_id),
        "userId": ObjectId(current_user["_id"]),
        "status": {"$ne": "completed"} # Prepare to prevent deleting completed? OR allow it.
    })
    
    if result.deleted_count == 0:
        # Check if it existed but was completed
        quiz = await math_quizzes_collection.find_one({"_id": ObjectId(quiz_id)})
        if quiz and quiz.get("status") == "completed":
             raise HTTPException(status_code=400, detail="Cannot delete completed quiz")
        raise HTTPException(status_code=404, detail="Quiz not found or cannot be deleted")
        
    return {"success": True}
