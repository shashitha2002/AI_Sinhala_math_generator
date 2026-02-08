from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List

class UserProfile(BaseModel):
    school: Optional[str] = None
    district: Optional[str] = None
    grade: Optional[str] = None
    studentId: Optional[str] = None

class UserSignUp(BaseModel):
    username: str
    email: EmailStr
    password: str
    profile: Optional[UserProfile] = None

class UserResponse(BaseModel):
    id: str = Field(..., alias="_id")
    username: str
    email: str
    role: str = "student"
    profile: Optional[UserProfile] = None
    performance: Optional[dict] = None

    class Config:
        populate_by_name = True

class Token(BaseModel):
    access_token: str
    token_type: str
    user: Optional[UserResponse] = None