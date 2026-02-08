from fastapi import APIRouter, HTTPException, Depends, status, Query
from typing import List, Optional
from app.database import forum_posts_collection, users_collection
from app.models.forum import ForumPost, Comment, ForumTopic
from app.models.utils import PyObjectId
from datetime import datetime
from bson import ObjectId

router = APIRouter(prefix="/forum", tags=["Forum"])

@router.get("/posts", response_model=dict)
async def get_posts(
    topic: Optional[ForumTopic] = None,
    search: Optional[str] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50)
):
    query = {}
    if topic:
        query["topic"] = topic
    if search:
        query["$or"] = [
            {"title": {"$regex": search, "$options": "i"}},
            {"content": {"$regex": search, "$options": "i"}}
        ]

    cursor = forum_posts_collection.find(query).sort("createdAt", -1).skip((page - 1) * limit).limit(limit)
    posts = await cursor.to_list(length=limit)
    
    # Manually handle ObjectId to string for the response if needed, 
    # but Pydantic Config populate_by_name and json_encoders should handle it if using response_model=List[ForumPost]
    # Here we return a dict with metadata
    
    total = await forum_posts_collection.count_documents(query)
    
    # Convert _id to id for each post
    for post in posts:
        post["_id"] = str(post["_id"])
        if "userId" in post:
            post["userId"] = str(post["userId"])
        for comment in post.get("comments", []):
            if "_id" in comment:
                comment["_id"] = str(comment["_id"])
            if "userId" in comment:
                comment["userId"] = str(comment["userId"])
    
    return {
        "success": True,
        "posts": posts,
        "totalPages": (total + limit - 1) // limit,
        "currentPage": page
    }

@router.post("/posts", status_code=201)
async def create_post(post: ForumPost):
    post_dict = post.dict(by_alias=True, exclude_none=True)
    if "_id" in post_dict:
        del post_dict["_id"]
    
    result = await forum_posts_collection.insert_one(post_dict)
    return {"success": True, "id": str(result.inserted_id)}

@router.get("/posts/{post_id}")
async def get_post(post_id: str):
    if not ObjectId.is_valid(post_id):
        raise HTTPException(status_code=400, detail="Invalid post ID")
    
    post = await forum_posts_collection.find_one({"_id": ObjectId(post_id)})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Increment views
    await forum_posts_collection.update_one(
        {"_id": ObjectId(post_id)},
        {"$inc": {"views": 1}}
    )
    
    post["_id"] = str(post["_id"])
    post["userId"] = str(post["userId"])
    for comment in post.get("comments", []):
        comment["userId"] = str(comment["userId"])
        if "_id" in comment:
            comment["_id"] = str(comment["_id"])
            
    return {"success": True, "post": post}

@router.post("/posts/{post_id}/comments")
async def add_comment(post_id: str, comment: Comment):
    if not ObjectId.is_valid(post_id):
        raise HTTPException(status_code=400, detail="Invalid post ID")
    
    comment_dict = comment.dict(by_alias=True, exclude_none=True)
    if "_id" in comment_dict:
        del comment_dict["_id"]
    comment_dict["_id"] = ObjectId()
    
    result = await forum_posts_collection.update_one(
        {"_id": ObjectId(post_id)},
        {"$push": {"comments": comment_dict}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Post not found")
        
    return {"success": True, "message": "Comment added"}

@router.post("/posts/{post_id}/like")
async def like_post(post_id: str, user_id: str):
    if not ObjectId.is_valid(post_id) or not ObjectId.is_valid(user_id):
        raise HTTPException(status_code=400, detail="Invalid ID")
    
    post = await forum_posts_collection.find_one({"_id": ObjectId(post_id)})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    user_oid = ObjectId(user_id)
    likes = post.get("likes", [])
    
    if user_oid in likes:
        # Unlike
        await forum_posts_collection.update_one(
            {"_id": ObjectId(post_id)},
            {"$pull": {"likes": user_oid}}
        )
        return {"success": True, "liked": False}
    else:
        # Like
        await forum_posts_collection.update_one(
            {"_id": ObjectId(post_id)},
            {"$push": {"likes": user_oid}}
        )
        return {"success": True, "liked": True}
