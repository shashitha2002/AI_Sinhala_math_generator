from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime
from app.models.utils import PyObjectId
from enum import Enum

class BadgeCriteriaType(str, Enum):
    quiz_completion = "quiz_completion"
    quiz_score = "quiz_score"
    topic_mastery = "topic_mastery"
    model_paper_score = "model_paper_score"
    time_spent = "time_spent"
    forum_participation = "forum_participation"
    streak = "streak"
    total_quizzes = "total_quizzes"
    game_achievement = "game_achievement"

class BadgeCondition(str, Enum):
    equals = "equals"
    greater_than = "greater_than"
    less_than = "less_than"
    greater_equal = "greater_equal"
    less_equal = "less_equal"
    contains = "contains"
    all_cond = "all"

class BadgeCriteria(BaseModel):
    type: BadgeCriteriaType
    condition: BadgeCondition
    value: Any
    topic: Optional[str] = None
    quizType: Optional[str] = None

class BadgeCategory(str, Enum):
    quiz = "quiz"
    forum = "forum"
    time = "time"
    streak = "streak"
    mastery = "mastery"
    achievement = "achievement"
    game = "game"

class BadgeRarity(str, Enum):
    common = "common"
    uncommon = "uncommon"
    rare = "rare"
    epic = "epic"
    legendary = "legendary"

class Badge(BaseModel):
    id: Optional[PyObjectId] = Field(None, alias="_id")
    badgeId: str
    name: str
    description: str
    icon: str = "🏆"
    category: BadgeCategory
    rarity: BadgeRarity = BadgeRarity.common
    criteria: List[BadgeCriteria]
    isActive: bool = True
    createdAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        json_encoders = {PyObjectId: str}
