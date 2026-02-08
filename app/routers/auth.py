from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from app.database import users_collection
from app.models.auth import UserSignUp, Token, UserResponse
from app.auth_utils import get_password_hash, verify_password, create_access_token
from app.dependencies import get_current_user
from bson import ObjectId

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/signup", status_code=201)
async def sign_up(user: UserSignUp):
    if await users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    new_user = {
        "username": user.username, 
        "email": user.email, 
        "password": get_password_hash(user.password),
        "role": "student",
        "profile": user.profile.dict() if user.profile else {
            "school": "",
            "district": "",
            "grade": "O-Level",
            "studentId": ""
        },
        "performance": {
            "totalQuizzes": 0,
            "averageScore": 0,
            "totalTimeSpent": 0,
            "badges": [],
            "achievements": []
        }
    }
    result = await users_collection.insert_one(new_user)
    
    # Generate token for auto-login
    access_token = create_access_token(data={"sub": user.email})
    
    user_id = str(result.inserted_id)
    return {
        "message": "User created",
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "_id": user_id,
            "username": new_user["username"],
            "email": new_user["email"],
            "role": new_user["role"],
            "profile": new_user["profile"]
        }
    }

@router.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await users_collection.find_one({"email": form_data.username})
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": user["email"]})
    
    # Prepare user response
    user["_id"] = str(user["_id"])
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user": user
    }

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    user = current_user.copy()
    user["_id"] = str(user["_id"])
    return user