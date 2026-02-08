from fastapi import APIRouter, HTTPException
from typing import List, Optional
from app.database import users_collection, performance_collection, quizzes_collection
from app.models.progress import UserPerformance, UserTopicMastery
from app.dependencies import get_current_user, Depends
from bson import ObjectId
from datetime import datetime

router = APIRouter(prefix="/progress", tags=["Progress"])

@router.get("/summary")
async def get_progress_summary(current_user: dict = Depends(get_current_user)):
    user = current_user
    user_id = str(user["_id"])
    if not ObjectId.is_valid(user_id):
        raise HTTPException(status_code=400, detail="Invalid user ID")
    
    user = await users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    performance = user.get("performance", {
        "totalQuizzes": 0,
        "averageScore": 0,
        "totalTimeSpent": 0,
        "badges": []
    })
    
    profile = user.get("profile", {
        "syllabusTopics": []
    })
    
    return {
        "success": True,
        "performance": performance,
        "topics": profile.get("syllabusTopics", [])
    }

@router.get("/history")
async def get_performance_history(current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    if not ObjectId.is_valid(user_id):
        raise HTTPException(status_code=400, detail="Invalid user ID")
    
    cursor = performance_collection.find({"studentId": ObjectId(user_id)}).sort("timestamp", -1).limit(50)
    history = await cursor.to_list(length=50)
    
    for h in history:
        h["_id"] = str(h["_id"])
        h["studentId"] = str(h["studentId"])
        h["questionId"] = str(h["questionId"])
        if "quizId" in h:
            h["quizId"] = str(h["quizId"])
            
    return {"success": True, "history": history}
@router.get("/dashboard")
async def get_dashboard(current_user: dict = Depends(get_current_user)):
    user = current_user
    user_id = user["_id"]
    
    # 1. Basic Stats from User performance object
    performance = user.get("performance", {
        "totalQuizzes": 0,
        "averageScore": 0,
        "totalTimeSpent": 0,
        "badges": []
    })
    
    # 2. Recent Quizzes
    cursor = quizzes_collection.find({"studentId": user_id}).sort("timestamp", -1).limit(5)
    recent_quizzes_list = await cursor.to_list(length=5)
    
    formatted_recent_quizzes = []
    for q in recent_quizzes_list:
        formatted_recent_quizzes.append({
            "date": q.get("timestamp", datetime.utcnow()).isoformat(),
            "type": q.get("type", "quiz"),
            "score": q.get("score", 0),
            "topic": q.get("questions", [{}])[0].get("topic", "General") if q.get("questions") else "General"
        })
    
    # 3. Topic Performance (Mocked if not exists, or aggregated)
    # For now, let's use some dummy data if nothing exists to match frontend structure
    topic_performance = user.get("topicPerformance", [
        {"topic": "Algebra", "accuracy": 75},
        {"topic": "Geometry", "accuracy": 60},
        {"topic": "Arithmetic", "accuracy": 90}
    ])
    
    # 4. Streak (Mocked)
    current_streak = user.get("streak", 3)
    
    return {
        "success": True,
        "dashboard": {
            "totalQuizzes": performance.get("totalQuizzes", 0),
            "averageScore": performance.get("averageScore", 0),
            "currentStreak": current_streak,
            "badges": performance.get("badges", []),
            "recentQuizzes": formatted_recent_quizzes,
            "topicPerformance": topic_performance,
            "mastery": topic_performance
        }
    }

@router.get("/portfolio")
async def get_portfolio(current_user: dict = Depends(get_current_user)):
    user_id = current_user["_id"]
    
    # Portfolio can consist of best works, certificates (derived from badges), and summary stats
    performance = current_user.get("performance", {})
    
    # Get high scoring quizzes
    cursor = quizzes_collection.find({
        "studentId": user_id, 
        "score.percentage": {"$gte": 80}
    }).sort("score.percentage", -1).limit(10)
    
    best_quizzes = await cursor.to_list(length=10)
    formatted_best_quizzes = []
    for q in best_quizzes:
        formatted_best_quizzes.append({
            "topic": q.get("questions", [{}])[0].get("topic", "General"),
            "score": q.get("score", {}).get("percentage", 0),
            "date": q.get("timestamp", datetime.utcnow()).isoformat()
        })

    return {
        "success": True,
        "portfolio": {
            "achievements": performance.get("badges", []),
            "bestWork": formatted_best_quizzes,
            "totalStudyTime": performance.get("totalTimeSpent", 0),
            "joined": current_user.get("createdAt", datetime.utcnow()).isoformat()
        }
    }
