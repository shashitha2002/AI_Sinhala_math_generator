from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from datetime import datetime, timedelta
from app.dependencies import get_current_user
from app.database import db

router = APIRouter(prefix="/games", tags=["Games"])

# Collections
games_collection = db["games_activity"]

@router.get("/activity")
async def get_activity_log(days: int = 30, current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    
    # Mock data for now, ideally fetch from games_activity collection
    # In a real app, you'd aggregate quizzes + games + posts
    
    today = datetime.utcnow()
    activity_log = []
    
    # Generate some dummy activity for the last 'days' days
    for i in range(days):
        date = today - timedelta(days=i)
        # Random activity
        has_activity = (date.day % 3 == 0) or (date.day % 5 == 0)
        
        if has_activity:
            activity_log.append({
                "date": date.isoformat(),
                "quizzes": 1 if date.day % 3 == 0 else 0,
                "games": 1 if date.day % 5 == 0 else 0,
                "posts": 0
            })
    
    return {
        "success": True,
        "activityLog": activity_log
    }

@router.post("/play")
async def record_game_play(game_data: dict, current_user: dict = Depends(get_current_user)):
    user_id = current_user["_id"]
    
    record = {
        "userId": user_id,
        "gameId": game_data.get("gameId"),
        "score": game_data.get("score", 0),
        "timestamp": datetime.utcnow()
    }
    
    await games_collection.insert_one(record)
    
    return {"success": True, "message": "Game recorded"}

@router.get("/stats")
async def get_game_stats(current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    
    return {
        "success": True,
        "stats": {
            "totalGames": 15,
            "highScore": 500,
            "rank": "Novice"
        }
    }
