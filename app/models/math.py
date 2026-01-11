from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class DifficultyLevel(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"

class QuestionRequest(BaseModel):
    topic: str = Field(..., example="වාරික ගණනය")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.medium)
    num_questions: int = Field(default=5, ge=1, le=10)

class Question(BaseModel):
    question: str
    solution: str
    answer: str

class QuestionResponse(BaseModel):
    success: bool
    topic: str
    questions: List[Question]
    generation_time_seconds: float
    model_used: str