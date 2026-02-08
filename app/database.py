import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")

client = AsyncIOMotorClient(MONGO_URL)
db = client.sinhala_math_db

users_collection = db.users
generated_questions_collection = db.generated_questions
forum_posts_collection = db.forum_posts
quizzes_collection = db.quizzes
badge_collection = db.badges
performance_collection = db.performance
syllabus_topics_collection = db.syllabus_topics
quiz_answers_collection = db.quiz_answers
math_quizzes_collection = db.math_quizzes