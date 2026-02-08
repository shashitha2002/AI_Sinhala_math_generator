from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from app.models.utils import PyObjectId
from enum import Enum

class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"

class ForumTopic(str, Enum):
    algebra = "algebra"
    geometry = "geometry"
    trigonometry = "trigonometry"
    statistics = "statistics"
    calculus = "calculus"
    general = "general"

class Comment(BaseModel):
    id: Optional[PyObjectId] = Field(None, alias="_id")
    userId: PyObjectId
    content: str
    latexContent: Optional[str] = None
    sentiment: Sentiment = Sentiment.neutral
    isModerated: bool = False
    moderationScore: Optional[float] = None
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: Optional[datetime] = None
    likes: List[PyObjectId] = []

    class Config:
        populate_by_name = True
        json_encoders = {PyObjectId: str}

class ForumPost(BaseModel):
    id: Optional[PyObjectId] = Field(None, alias="_id")
    userId: PyObjectId
    title: str
    content: str
    latexContent: Optional[str] = None
    topic: ForumTopic
    tags: List[str] = []
    sentiment: Sentiment = Sentiment.neutral
    isModerated: bool = False
    moderationScore: Optional[float] = None
    moderationReason: Optional[str] = None
    comments: List[Comment] = []
    likes: List[PyObjectId] = []
    views: int = 0
    isResolved: bool = False
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: Optional[datetime] = None

    class Config:
        populate_by_name = True
        json_encoders = {PyObjectId: str}
