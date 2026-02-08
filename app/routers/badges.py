from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.database import badge_collection, users_collection
from app.models.badge import Badge
from bson import ObjectId
from app.routers.auth import get_current_user


DEFAULT_BADGES = [
    {"badgeId": "first_quiz", "name": "First Steps", "description": "Completed your first quiz", "rarity": "common", "icon": "target", "condition": {"type": "quiz_count", "value": 1}, "isActive": True},
    {"badgeId": "quiz_master_10", "name": "Quiz Enthusiast", "description": "Completed 10 quizzes", "rarity": "common", "icon": "file-text", "condition": {"type": "quiz_count", "value": 10}, "isActive": True},
    {"badgeId": "quiz_master_50", "name": "Quiz Master", "description": "Completed 50 quizzes", "rarity": "rare", "icon": "award", "condition": {"type": "quiz_count", "value": 50}, "isActive": True},
    {"badgeId": "quiz_master_100", "name": "Quiz Legend", "description": "Completed 100 quizzes", "rarity": "epic", "icon": "crown", "condition": {"type": "quiz_count", "value": 100}, "isActive": True},
    {"badgeId": "excellent_score", "name": "Excellent Performer", "description": "Achieved an average score of 80% or higher", "rarity": "uncommon", "icon": "star", "condition": {"type": "avg_score", "value": 80}, "isActive": True},
    {"badgeId": "perfect_score", "name": "Perfect Score", "description": "Achieved 100% on a quiz", "rarity": "rare", "icon": "100", "condition": {"type": "single_score", "value": 100}, "isActive": True},
    {"badgeId": "model_paper_80", "name": "Model Paper Expert", "description": "Scored 80% or higher on a model paper", "rarity": "uncommon", "icon": "file-check", "condition": {"type": "model_paper_score", "value": 80}, "isActive": True},
    {"badgeId": "model_paper_90", "name": "Model Paper Master", "description": "Scored 90% or higher on a model paper", "rarity": "rare", "icon": "graduation-cap", "condition": {"type": "model_paper_score", "value": 90}, "isActive": True},
    {"badgeId": "topic_master", "name": "Topic Master", "description": "Achieved 80% mastery in any topic", "rarity": "uncommon", "icon": "target", "condition": {"type": "topic_mastery", "value": 80}, "isActive": True},
    {"badgeId": "dedicated_learner", "name": "Dedicated Learner", "description": "Spent 10 hours (600 minutes) on the platform", "rarity": "common", "icon": "clock", "condition": {"type": "time_spent", "value": 600}, "isActive": True},
    {"badgeId": "time_champion", "name": "Time Champion", "description": "Spent 50 hours (3000 minutes) on the platform", "rarity": "rare", "icon": "hourglass", "condition": {"type": "time_spent", "value": 3000}, "isActive": True},
    {"badgeId": "consistent_7", "name": "Week Warrior", "description": "Maintained a 7-day learning streak", "rarity": "uncommon", "icon": "fire", "condition": {"type": "streak", "value": 7}, "isActive": True},
    {"badgeId": "consistent_30", "name": "Monthly Champion", "description": "Maintained a 30-day learning streak", "rarity": "epic", "icon": "sun", "condition": {"type": "streak", "value": 30}, "isActive": True},
    {"badgeId": "game_explorer", "name": "Game Explorer", "description": "Played your first game", "rarity": "common", "icon": "gamepad", "condition": {"type": "game_count", "value": 1}, "isActive": True},
    {"badgeId": "arcade_fan", "name": "Arcade Fan", "description": "Played 10 games", "rarity": "uncommon", "icon": "joystick", "condition": {"type": "game_count", "value": 10}, "isActive": True},
    {"badgeId": "high_scorer", "name": "High Scorer", "description": "Achieved a score of 1000 or more in a game", "rarity": "rare", "icon": "target-arrow", "condition": {"type": "game_score", "value": 1000}, "isActive": True},
    {"badgeId": "avg_score_45", "name": "Bronze Scholar", "description": "Achieved an average score above 45%", "rarity": "common", "icon": "medal", "condition": {"type": "avg_score", "value": 45}, "isActive": True},
    {"badgeId": "avg_score_55", "name": "Silver Scholar", "description": "Achieved an average score above 55%", "rarity": "uncommon", "icon": "medal", "condition": {"type": "avg_score", "value": 55}, "isActive": True},
    {"badgeId": "avg_score_65", "name": "Gold Scholar", "description": "Achieved an average score above 65%", "rarity": "rare", "icon": "medal", "condition": {"type": "avg_score", "value": 65}, "isActive": True},
    {"badgeId": "avg_score_75", "name": "Platinum Scholar", "description": "Achieved an average score above 75%", "rarity": "epic", "icon": "diamond", "condition": {"type": "avg_score", "value": 75}, "isActive": True}
]

async def initialize_default_badges():
    try:
        if await badge_collection.count_documents({}) == 0:
            await badge_collection.insert_many(DEFAULT_BADGES)
            print("Badges initialized successfully")
    except Exception as e:
        print(f"Error initializing badges: {e}")

router = APIRouter(prefix="/badges", tags=["Badges"])

@router.get("/all", response_model=List[dict])
async def get_all_badges():
    count = await badge_collection.count_documents({})
    if count == 0:
        await initialize_default_badges()
        
    badges = await badge_collection.find({"isActive": True}).to_list(length=100)
    for b in badges:
        b["_id"] = str(b["_id"])
    return badges

@router.get("/user/{user_id}")
async def get_user_badges(user_id: str):
    if not ObjectId.is_valid(user_id):
        raise HTTPException(status_code=400, detail="Invalid user ID")
    
    user = await users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_badge_ids = user.get("performance", {}).get("badges", [])
    
    badges = await badge_collection.find({"badgeId": {"$in": user_badge_ids}}).to_list(length=100)
    for b in badges:
        b["_id"] = str(b["_id"])
        
    return {"success": True, "badges": badges}

@router.get("/my-badges")
async def get_my_badges(current_user: dict = Depends(get_current_user)):
    # Ensure badges are initialized
    count = await badge_collection.count_documents({})
    if count == 0:
        await initialize_default_badges()
    
    user_badge_ids = current_user.get("performance", {}).get("badges", [])
    
    # Get all badges
    all_badges = await badge_collection.find({"isActive": True}).to_list(length=100)
    
    # Mark badges as earned or not
    for badge in all_badges:
        badge["_id"] = str(badge["_id"])
        badge["earned"] = badge["badgeId"] in user_badge_ids
        if badge["earned"]:
            # You could add earnedDate from user's performance data if available
            badge["earnedDate"] = current_user.get("performance", {}).get("badgeEarnedDates", {}).get(badge["badgeId"])
    
    # Separate earned and all badges
    earned_badges = [b for b in all_badges if b["earned"]]
    
    # Calculate stats
    total_earned = len(earned_badges)
    total_available = len(all_badges)
    progress = round((total_earned / total_available * 100) if total_available > 0 else 0, 2)
    
    return {
        "success": True,
        "allBadges": all_badges,
        "earnedBadges": earned_badges,
        "totalEarned": total_earned,
        "totalAvailable": total_available,
        "progress": progress
    }
