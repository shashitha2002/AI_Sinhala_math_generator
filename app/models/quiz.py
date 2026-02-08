from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from app.models.utils import PyObjectId
from enum import Enum

class QuizType(str, Enum):
    adaptive = "adaptive"
    model_paper = "model-paper"
    practice = "practice"

class QuizStatus(str, Enum):
    in_progress = "in-progress"
    completed = "completed"
    abandoned = "abandoned"

class IRTParameters(BaseModel):
    discrimination: float = 1.0
    difficulty: float = 0.0
    guessing: float = 0.25

class QuizQuestion(BaseModel):
    questionId: Optional[str] = None
    question: str
    options: List[str]
    correctAnswer: int
    explanation: Optional[str] = None
    topic: Optional[str] = None
    difficulty: str = "medium"
    marks: int = 1
    syllabusUnit: Optional[str] = None
    irtParameters: Optional[IRTParameters] = None

class QuizAnswer(BaseModel):
    questionId: str
    selectedAnswer: int
    isCorrect: bool
    timeSpent: int # in seconds
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class QuizScore(BaseModel):
    total: int
    obtained: int
    percentage: float

class AdaptiveParams(BaseModel):
    initialDifficulty: float = 0.0
    adjustments: List[dict] = []

class Quiz(BaseModel):
    id: Optional[PyObjectId] = Field(None, alias="_id")
    userId: PyObjectId
    type: QuizType = QuizType.adaptive
    questions: List[QuizQuestion]
    answers: List[QuizAnswer] = []
    score: Optional[QuizScore] = None
    timeLimit: Optional[int] = None # in minutes
    timeStarted: datetime = Field(default_factory=datetime.utcnow)
    timeCompleted: Optional[datetime] = None
    status: QuizStatus = QuizStatus.in_progress
    adaptiveParams: Optional[AdaptiveParams] = None
    createdAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        json_encoders = {PyObjectId: str}
