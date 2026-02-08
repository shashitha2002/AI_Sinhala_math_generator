from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from app.models.utils import PyObjectId

class PerformanceData(BaseModel):
    id: Optional[PyObjectId] = Field(None, alias="_id")
    studentId: PyObjectId
    questionId: PyObjectId
    topicId: str
    isCorrect: bool
    timeTaken: int # in seconds
    attemptsCount: int = 1
    studentProficiency: float
    questionDifficulty: float
    probabilityCorrect: float
    quizId: Optional[PyObjectId] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        json_encoders = {PyObjectId: str}

class UserTopicMastery(BaseModel):
    topic: str
    mastery: float = 0.0

class UserPerformance(BaseModel):
    totalQuizzes: int = 0
    averageScore: float = 0.0
    totalTimeSpent: float = 0.0 # in minutes
    badges: List[str] = []
    achievements: List[dict] = []

class UserProfile(BaseModel):
    syllabusTopics: List[UserTopicMastery] = []
