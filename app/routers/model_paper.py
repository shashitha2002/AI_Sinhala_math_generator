"""
Model Paper Generation Router - Separate APIs for each question type
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import time

from app.models.model_paper_generator import ModelPaperGenerator
from app.dependencies import get_model_paper_generator, get_current_user

router = APIRouter(
    prefix="/model-paper",
    tags=["Model Paper Generation"]
)


# ==================== Pydantic Models ====================

class GenerateShortAnswerRequest(BaseModel):
    count: int = Field(default=5, ge=1, le=10, description="Number of questions (1-10)")
    topics: Optional[List[str]] = Field(default=None, description="Optional list of topics")


class GenerateStructuredRequest(BaseModel):
    count: int = Field(default=3, ge=1, le=5, description="Number of questions (1-5)")
    topics: Optional[List[str]] = Field(default=None, description="Optional list of topics")


class GenerateEssayRequest(BaseModel):
    count: int = Field(default=2, ge=1, le=5, description="Number of questions (1-5)")
    topics: Optional[List[str]] = Field(default=None, description="Optional list of topics")


class AnswerStep(BaseModel):
    description: str
    value: str


class SubQuestion(BaseModel):
    sub_question_label: str
    sub_question: str
    answer_steps: List[AnswerStep] = []
    answer: Optional[str] = None


class ShortAnswerQuestion(BaseModel):
    question_number: int
    question: str
    topics: List[str]
    answer_steps: List[AnswerStep]
    final_answer: Optional[str] = None


class StructuredQuestion(BaseModel):
    question_number: int
    question: str
    topics: List[str]
    sub_questions: List[SubQuestion]


class EssayQuestion(BaseModel):
    question_number: int
    question: str
    topics: List[str]
    sub_questions: List[SubQuestion]


class GenerationResponse(BaseModel):
    success: bool
    type: str
    questions: List[Dict]
    count: int
    requested: int
    topics_used: List[str]
    generation_time_seconds: float


# ==================== Status & Initialize ====================

@router.get("/status")
async def get_status(
    generator: ModelPaperGenerator = Depends(get_model_paper_generator)
):
    """Get current generator status"""
    return {
        "initialized": True,
        "past_papers_loaded": generator.past_papers_loaded,
        "available_topics": generator.available_topics,
        "total_topics": len(generator.available_topics)
    }


@router.post("/initialize")
async def initialize_generator(
    past_papers_path: str = "data/extracted_text/model_paper_questions.json",
    generator: ModelPaperGenerator = Depends(get_model_paper_generator)
):
    """Initialize and load past papers"""
    try:
        success = generator.load_past_paper_questions(past_papers_path)
        
        if not success:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to load past papers from {past_papers_path}"
            )
        
        return {
            "success": True,
            "message": "Model Paper Generator initialized",
            "statistics": generator.get_statistics()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/topics")
async def get_available_topics(
    generator: ModelPaperGenerator = Depends(get_model_paper_generator)
):
    """Get list of available topics"""
    if not generator.past_papers_loaded:
        raise HTTPException(
            status_code=400,
            detail="Past papers not loaded. Call /model-paper/initialize first."
        )
    
    stats = generator.get_statistics()
    
    return {
        "available_topics": stats["available_topics"],
        "total_topics": len(stats["available_topics"]),
        "questions_by_topic": stats["questions_by_topic"],
        "questions_by_type": stats["questions_by_type"]
    }


# ==================== SHORT ANSWER API ====================

@router.post("/generate/short-answer", response_model=GenerationResponse)
async def generate_short_answer(
    request: GenerateShortAnswerRequest = GenerateShortAnswerRequest(),
    generator: ModelPaperGenerator = Depends(get_model_paper_generator),
    current_user: dict = Depends(get_current_user)
):
    """
    Generate short answer questions.
    
    ### Features:
    - Simple, direct questions
    - 2-4 solution steps
    - Topics: algebra, percentages, interest, etc.
    
    ### Output Format:
    Each question has:
    - `question`: The question text
    - `answer_steps`: List of {description, value} pairs
    - `final_answer`: The final answer
    """
    if not generator.past_papers_loaded:
        raise HTTPException(
            status_code=400,
            detail="Past papers not loaded. Call /model-paper/initialize first."
        )
    
    try:
        result = generator.generate_short_answer_questions(
            count=request.count,
            topics=request.topics
        )
        
        return GenerationResponse(
            success=True,
            type="short_answer",
            questions=result["questions"],
            count=result["count"],
            requested=result["requested"],
            topics_used=result["topics_used"],
            generation_time_seconds=result["generation_time_seconds"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== STRUCTURED API ====================

@router.post("/generate/structured", response_model=GenerationResponse)
async def generate_structured(
    request: GenerateStructuredRequest = GenerateStructuredRequest(),
    generator: ModelPaperGenerator = Depends(get_model_paper_generator),
    current_user: dict = Depends(get_current_user)
):
    """
    Generate structured questions with sub-questions.
    
    ### Features:
    - Main context/scenario
    - 3-5 related sub-questions (අ, ආ, ඇ, ඈ, ඉ)
    - Each sub-question has solution steps
    - Sub-questions build on each other
    
    ### Output Format:
    Each question has:
    - `question`: Main context/scenario
    - `sub_questions`: List of sub-questions with:
      - `sub_question_label`: (අ), (ආ), etc.
      - `sub_question`: The sub-question text
      - `answer_steps`: Solution steps
      - `answer`: Final answer for this part
    """
    if not generator.past_papers_loaded:
        raise HTTPException(
            status_code=400,
            detail="Past papers not loaded. Call /model-paper/initialize first."
        )
    
    try:
        result = generator.generate_structured_questions(
            count=request.count,
            topics=request.topics
        )
        
        return GenerationResponse(
            success=True,
            type="structured",
            questions=result["questions"],
            count=result["count"],
            requested=result["requested"],
            topics_used=result["topics_used"],
            generation_time_seconds=result["generation_time_seconds"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ESSAY TYPE API ====================

@router.post("/generate/essay", response_model=GenerationResponse)
async def generate_essay(
    request: GenerateEssayRequest = GenerateEssayRequest(),
    generator: ModelPaperGenerator = Depends(get_model_paper_generator),
    current_user: dict = Depends(get_current_user)
):
    """
    Generate essay type questions with real-life scenarios.
    
    ### Features:
    - Detailed real-life scenario (3-5 sentences)
    - 4-6 progressive sub-questions (i, ii, iii, iv, v)
    - Uses Sinhala names and Sri Lankan context
    - Final question often asks to compare/conclude
    
    ### Output Format:
    Each question has:
    - `question`: Detailed real-life scenario
    - `sub_questions`: List of sub-questions with:
      - `sub_question_label`: (i), (ii), etc.
      - `sub_question`: The sub-question text
      - `answer_steps`: Solution steps
      - `answer`: Final answer for this part
    
    ### Example Scenario Types:
    - Banking/Interest: "සුමන රුපියල් 100000 බැංකුවක තැන්පත් කරයි..."
    - Stock Market: "සමාගමක කොටස් 10000ක් නිකුත් කර ඇත..."
    - Daily Life: "කමල් තම නිවස මසකට රුපියල් 8000 බැගින් බදු දෙයි..."
    """
    if not generator.past_papers_loaded:
        raise HTTPException(
            status_code=400,
            detail="Past papers not loaded. Call /model-paper/initialize first."
        )
    
    try:
        result = generator.generate_essay_questions(
            count=request.count,
            topics=request.topics
        )
        
        return GenerationResponse(
            success=True,
            type="essay_type",
            questions=result["questions"],
            count=result["count"],
            requested=result["requested"],
            topics_used=result["topics_used"],
            generation_time_seconds=result["generation_time_seconds"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== SAMPLE OUTPUTS ====================

@router.get("/sample/short-answer")
async def sample_short_answer():
    """Sample output format for short answer questions"""
    return {
        "success": True,
        "type": "short_answer",
        "questions": [
            {
                "question_number": 1,
                "question": "සුළු කරන්න: (2/3x) + (5/6x) - (7/12x)",
                "topics": ["වීජීය භාග"],
                "answer_steps": [
                    {"description": "පොදු හරය සොයන්න", "value": "12x"},
                    {"description": "භාග සමාන කරන්න", "value": "8/12x + 10/12x - 7/12x"},
                    {"description": "සරල කරන්න", "value": "11/12x"}
                ],
                "final_answer": "11/12x"
            },
            {
                "question_number": 2,
                "question": "සාධක සොයන්න: 2x² - 18",
                "topics": ["වර්ගජ ප්‍රකාශනවල සාධක"],
                "answer_steps": [
                    {"description": "පොදු සාධකය ගන්න", "value": "2(x² - 9)"},
                    {"description": "වර්ග අන්තරය භාවිතා කරන්න", "value": "2(x-3)(x+3)"}
                ],
                "final_answer": "2(x-3)(x+3)"
            },
            {
                "question_number": 3,
                "question": "10^0.6375 = 4.34 ලෙස ගෙන lg 43.4 හි අගය සොයන්න",
                "topics": ["ලඝුගණක"],
                "answer_steps": [
                    {"description": "43.4 ලියන්න", "value": "43.4 = 4.34 × 10 = 10^0.6375 × 10^1"},
                    {"description": "lg 43.4 ගණනය", "value": "0.6375 + 1 = 1.6375"}
                ],
                "final_answer": "1.6375"
            }
        ],
        "count": 3,
        "requested": 3,
        "topics_used": ["වීජීය භාග", "වර්ගජ ප්‍රකාශනවල සාධක", "ලඝුගණක"],
        "generation_time_seconds": 12.5
    }


@router.get("/sample/structured")
async def sample_structured():
    """Sample output format for structured questions"""
    return {
        "success": True,
        "type": "structured",
        "questions": [
            {
                "question_number": 1,
                "question": "ජනක තම මාසික වැටුප රුපියල් 100000 කට වඩා වැඩි වූ විට එම වැඩිවන මුදලට 6% ක් ආදායම් බදු ලෙස ගෙවයි. එක්තරා මාසයකදී බදු ගෙවීමෙන් පසු ඔහුට ලැබුණු මුදලින් 1/6 ක් ඔහු ආහාර සඳහා වෙන් කරයි. ඉතිරි මුදලින් 3/5 ක් ඔහුගේ වෙනත් වියදම් සඳහා වෙන් කරයි.",
                "topics": ["ප්‍රතිශත", "භාග"],
                "sub_questions": [
                    {
                        "sub_question_label": "(අ)",
                        "sub_question": "ජනකට ලැබුණු මුදලින් 1/6 ක් ආහාර සඳහා වෙන් කළ පසු ඔහුට එම මුදලින් කවර භාගයක් ඉතිරි වේ ද?",
                        "answer_steps": [
                            {"description": "ඉතිරි භාගය ගණනය", "value": "1 - 1/6 = 5/6"}
                        ],
                        "answer": "5/6"
                    },
                    {
                        "sub_question_label": "(ආ)",
                        "sub_question": "ආහාර සහ වෙනත් වියදම් සඳහා මුදල් වෙන් කළ පසු ජනකට ඉතිරි වන්නේ ලැබූ මුදලින් කවර භාගයක් ද?",
                        "answer_steps": [
                            {"description": "වෙනත් වියදම් භාගය", "value": "(5/6) × (3/5) = 3/6 = 1/2"},
                            {"description": "ඉතිරි භාගය", "value": "1 - (1/6 + 1/2) = 1 - 4/6 = 2/6 = 1/3"}
                        ],
                        "answer": "1/3"
                    },
                    {
                        "sub_question_label": "(ඇ)",
                        "sub_question": "ඔහුට දැන් ඉතිරිවන මුදල රුපියල් 39600 ක් නම් බදු ගෙවීමෙන් පසු ඔහුට ලැබුණු මුදලත් ආහාර සඳහා වෙන් කළ මුදලත් වෙන වෙනම සොයන්න.",
                        "answer_steps": [
                            {"description": "බදු ගෙවීමෙන් පසු ලැබුනු මුදල", "value": "39600 × 3 = රු. 118800"},
                            {"description": "ආහාර සඳහා වෙන් කල මුදල", "value": "118800 × (1/6) = රු. 19800"}
                        ],
                        "answer": "බදු ගෙවීමෙන් පසු: රු. 118800, ආහාර සඳහා: රු. 19800"
                    },
                    {
                        "sub_question_label": "(ඈ)",
                        "sub_question": "බදු ගෙවීමට පෙර ඔහුගේ වැටුප කීයද?",
                        "answer_steps": [
                            {"description": "බදු ගෙවූ මුදල", "value": "118800 - 100000 = 18800 යනු 94% ට සමානයි"},
                            {"description": "බදු ගෙවීමට පෙර අමතර මුදල", "value": "18800 × (100/94) = රු. 20000"},
                            {"description": "මුළු වැටුප", "value": "100000 + 20000 = රු. 120000"}
                        ],
                        "answer": "රු. 120000"
                    },
                    {
                        "sub_question_label": "(ඉ)",
                        "sub_question": "යම් අවස්ථාවක බදු අයකර ගැනීමේ සීමාව ඉහළ දැමීම නිසා ජනක ආදායම් බදු ගෙවීමෙන් නිදහස් වේ නම් සහ ඔහු ආහාර සඳහා මුලදී වියදම් කළ මුදල වෙනස් නොවී පවතී නම් දැන් ඔහු ආහාර සඳහා වියදම් කරන මුදල වැටුපෙන් කවර ප්‍රතිශතයක් ද?",
                        "answer_steps": [
                            {"description": "ප්‍රතිශතය ගණනය", "value": "(19800 / 120000) × 100% = 16.5%"}
                        ],
                        "answer": "16.5%"
                    }
                ]
            }
        ],
        "count": 1,
        "requested": 1,
        "topics_used": ["ප්‍රතිශත", "භාග"],
        "generation_time_seconds": 25.3
    }


@router.get("/sample/essay")
async def sample_essay():
    """Sample output format for essay type questions"""
    return {
        "success": True,
        "type": "essay_type",
        "questions": [
            {
                "question_number": 1,
                "question": "එකක් රුපියල් 84000 බැගින් වටිනා රූපවාහිනී තොගයක් විකිණීමට තිබේ. රුවිනි එක් රූපවාහිනියක් මිලදී ගන්නා ආකාරයත් මානෙල් තවත් රූපවාහිනියක් මිලදී ගන්නා ආකාරයත් පහත දැක්වේ. රුවිනි: මූල්‍ය ආයතනයකින් රුපියල් 84000 ක් වාර්ෂික සුළු පොලියට අවුරුද්දකට ණයට ගෙන රූපවාහිනිය මිලදී ගනියි. අවුරුද්ද අවසානයේ රුපියල් 10920 ක පොලියක් සමග ණය මුදල ගෙවා ණයෙන් නිදහස් වෙයි. මානෙල්: කුලී කිණීමේ පදනම මත සමාන මාසික වාරික 12 කින් පොලියත් සමග මුදල් ගෙවීමට රූපවාහිනිය මිලදී ගනියි. මෙහි පොලිය ගණනය කරනු ලබන්නේ හීනවන ශේෂ ක්‍රමයට ය. අවුරුද්දකදී වාරික ගෙවා අවසන් වන විට මුළු පොලිය ලෙස රුවිනි ගෙවන පොලියම වන රුපියල් 10920 ක් ගෙවයි.",
                "topics": ["පොලිය", "කුලී මිලදී ගැනීම"],
                "sub_questions": [
                    {
                        "sub_question_label": "(i)",
                        "sub_question": "රුව��නි සඳහා වාර්ෂික පොලී අනුපාතිකය කීයද?",
                        "answer_steps": [
                            {"description": "පොලී අනුපාතිකය සූත්‍රය", "value": "(පොලිය / මුදල) × 100"},
                            {"description": "ගණනය", "value": "(10920 / 84000) × 100 = 13%"}
                        ],
                        "answer": "13%"
                    },
                    {
                        "sub_question_label": "(ii)",
                        "sub_question": "මානෙල් සඳහා මාස ඒකක ගණන කීයද?",
                        "answer_steps": [
                            {"description": "මාස ඒකක සූත්‍රය", "value": "n(n+1)/2"},
                            {"description": "ගණනය", "value": "12(12+1)/2 = 12 × 13/2 = 78"}
                        ],
                        "answer": "78"
                    },
                    {
                        "sub_question_label": "(iii)",
                        "sub_question": "එක් මාස ඒකකයකට පොලිය කීයද?",
                        "answer_steps": [
                            {"description": "එක් ඒකකයකට පොලිය", "value": "මුළු පොලිය / මාස ඒකක ගණන"},
                            {"description": "ගණනය", "value": "10920 / 78 = රු. 140"}
                        ],
                        "answer": "රු. 140"
                    },
                    {
                        "sub_question_label": "(iv)",
                        "sub_question": "එක් වාරිකයක ණය මුදල (ප්‍රාග්ධනය) කීයද?",
                        "answer_steps": [
                            {"description": "වාරිකයක ණය මුදල", "value": "84000 / 12 = රු. 7000"}
                        ],
                        "answer": "රු. 7000"
                    },
                    {
                        "sub_question_label": "(v)",
                        "sub_question": "මානෙල්ගේ වාර්ෂික පොලී අනුපාතිකය සොයා, කුලී කිණීමේ ක්‍රමයේදී අය කරනු ලබන වාර්ෂික පොලී අනුපාතිකය මූල්‍ය ආයතනය අය කරනු ලබන වාර්ෂික පොලී අනුපාතිකයට වඩා වැඩි බව පෙන්වන්න.",
                        "answer_steps": [
                            {"description": "පොලී සූත්‍රය", "value": "පොලිය = (ප්‍රාග්ධනය × R × T) / (100 × 12)"},
                            {"description": "R ගණනය", "value": "140 = (7000 × R × 1) / (100 × 12)"},
                            {"description": "R සොයන්න", "value": "R = (140 × 1200) / 7000 = 24%"},
                            {"description": "සංසන්දනය", "value": "24% > 13%"}
                        ],
                        "answer": "මානෙල්ගේ පොලී අනුපාතිකය (24%) රුවිනිගේ පොලී අනුපාතිකයට (13%) වඩා වැඩි බැවින් ප්‍රකාශය සත්‍ය වේ."
                    }
                ]
            },
            {
                "question_number": 2,
                "question": "අමලා සහ සුමනා නිවාඩු කාලය තුළදී එක්තරා නවකතාවක් කියවීමට තීරණය කරති. අමලා පළමුවන දිනයේදී පිටු 20 ක් කියවන අතර ඉන්පසු සෑම දිනකම ඇය ඊට පෙර දින කියවූ පිටු සංඛ්‍යාවට වඩා පිටු තුනක් වැඩියෙන් කියවයි.",
                "topics": ["සමාන්තර ශ්‍රේණි"],
                "sub_questions": [
                    {
                        "sub_question_label": "(i)",
                        "sub_question": "පළමුවන, දෙවන සහ තුන්වන දිනවලදී අමලා කියවන පිටු සංඛ්‍යා පිළිවෙළින් ලියා දක්වන්න.",
                        "answer_steps": [
                            {"description": "පිටු ගණන", "value": "20, 23, 26"}
                        ],
                        "answer": "20, 23, 26"
                    },
                    {
                        "sub_question_label": "(ii)",
                        "sub_question": "අමලා 16 වන දිනයේදී පිටු කීයක් කියවයි ද?",
                        "answer_steps": [
                            {"description": "Tₙ සූත්‍රය", "value": "Tₙ = a + (n-1)d"},
                            {"description": "T₁₆ ගණනය", "value": "T₁₆ = 20 + (16-1)×3 = 20 + 45 = 65"}
                        ],
                        "answer": "65"
                    },
                    {
                        "sub_question_label": "(iii)",
                        "sub_question": "ඇය 16 වන දිනයේදී නවකතාව මුළුමනින්ම කියවා නිම කරයි නම් නවකතාව පිටු කීයකින් සමන්විත වේ ද?",
                        "answer_steps": [
                            {"description": "Sₙ සූත්‍රය", "value": "Sₙ = (n/2)(a + l)"},
                            {"description": "S₁₆ ගණනය", "value": "S₁₆ = (16/2)(20 + 65) = 8 × 85 = 680"}
                        ],
                        "answer": "680"
                    },
                    {
                        "sub_question_label": "(iv)",
                        "sub_question": "සුමනා එම නවකතාව කියවීම ආරම්භ කළ පළමුවන දිනයෙන් පසු සෑම දිනකම ඊට පෙර දින කියවූ පිටු සංඛ්‍යාවට වඩා පිටු 4 ක් වැඩියෙන් කියවයි නම් සහ ඇය දින 17 කදී නවකතාව මුළුමනින්ම කියවා නිම කරයි නම් ඇය පළමුවන දිනයේ නවකතා පොතෙහි පිටු කීයක් කියවයි ද?",
                        "answer_steps": [
                            {"description": "Sₙ සූත්‍රය", "value": "Sₙ = (n/2)(2a + (n-1)d)"},
                            {"description": "සමීකරණය", "value": "680 = (17/2)(2a + 16×4)"},
                            {"description": "විසඳීම", "value": "680 = 8.5(2a + 64), 80 = 2a + 64, a = 8"}
                        ],
                        "answer": "8"
                    },
                    {
                        "sub_question_label": "(v)",
                        "sub_question": "මේ දෙදෙනාම එකම දිනයකදී නවකතාව කියවීම ආරම්භ කළේ නම් ඔවුන් දෙදෙනා එකම පිටු සංඛ්‍යාවක් කියවන්නේ කුමන දිනයේ ද?",
                        "answer_steps": [
                            {"description": "අමලාගේ n වන දින", "value": "20 + (n-1)×3"},
                            {"description": "සුමනාගේ n වන දින", "value": "8 + (n-1)×4"},
                            {"description": "සමීකරණය", "value": "20 + 3n - 3 = 8 + 4n - 4"},
                            {"description": "විසඳීම", "value": "17 + 3n = 4 + 4n, n = 13"}
                        ],
                        "answer": "13 වන දිනයේ"
                    }
                ]
            }
        ],
        "count": 2,
        "requested": 2,
        "topics_used": ["පොලිය", "කුලී මිලදී ගැනීම", "සමාන්තර ශ්‍රේණි"],
        "generation_time_seconds": 45.7
    }


# ==================== GENERATE FULL PAPER (COMBINED) ====================

class GenerateFullPaperRequest(BaseModel):
    short_answer_count: int = Field(default=25, ge=1, le=25)
    structured_count: int = Field(default=5, ge=1, le=10)
    essay_count: int = Field(default=10, ge=1, le=10)


@router.post("/generate/full-paper")
async def generate_full_paper(
    request: GenerateFullPaperRequest = GenerateFullPaperRequest(),
    generator: ModelPaperGenerator = Depends(get_model_paper_generator),
    current_user: dict = Depends(get_current_user)
):
    """
    Generate a complete model paper by calling all three APIs.
    
    ⚠️ This is a long-running operation (5-10 minutes)
    
    For faster results, use individual endpoints:
    - POST /model-paper/generate/short-answer
    - POST /model-paper/generate/structured
    - POST /model-paper/generate/essay
    """
    if not generator.past_papers_loaded:
        raise HTTPException(
            status_code=400,
            detail="Past papers not loaded. Call /model-paper/initialize first."
        )
    
    start_time = time.time()
    all_topics_used = set()
    
    try:
        # Generate short answer questions
        print("\n📝 Generating short answer questions...")
        short_answer_result = generator.generate_short_answer_questions(
            count=request.short_answer_count
        )
        all_topics_used.update(short_answer_result["topics_used"])
        
        # Generate structured questions
        print("\n📋 Generating structured questions...")
        structured_result = generator.generate_structured_questions(
            count=request.structured_count
        )
        all_topics_used.update(structured_result["topics_used"])
        
        # Generate essay questions
        print("\n📝 Generating essay questions...")
        essay_result = generator.generate_essay_questions(
            count=request.essay_count
        )
        all_topics_used.update(essay_result["topics_used"])
        
        total_time = round(time.time() - start_time, 2)
        
        return {
            "success": True,
            "paper_id": f"MP_{int(time.time())}",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "questions": {
                "short_answer": short_answer_result["questions"],
                "structured": structured_result["questions"],
                "essay_type": essay_result["questions"]
            },
            "summary": {
                "short_answer": {
                    "requested": request.short_answer_count,
                    "generated": short_answer_result["count"]
                },
                "structured": {
                    "requested": request.structured_count,
                    "generated": structured_result["count"]
                },
                "essay_type": {
                    "requested": request.essay_count,
                    "generated": essay_result["count"]
                }
            },
            "topics_used": list(all_topics_used),
            "generation_time_seconds": total_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== HEALTH CHECK ====================

@router.get("/health")
async def health_check():
    """Health check for model paper generation service"""
    return {
        "status": "healthy",
        "service": "model-paper-generator",
        "endpoints": [
            "/model-paper/generate/short-answer",
            "/model-paper/generate/structured",
            "/model-paper/generate/essay",
            "/model-paper/generate/full-paper"
        ]
    }