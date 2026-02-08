"""
Model Paper Generator - Separated by Question Type
Generates O/L Mathematics questions with proper structure
"""

import json
import os
import time
import random
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import google.generativeai as genai


class QuestionType(Enum):
    SHORT_ANSWER = "short_answer"
    STRUCTURED = "structured"
    ESSAY_TYPE = "essay_type"


@dataclass
class GenerationConfig:
    """Configuration for question generation"""
    count: int = 5
    api_delay: float = 4.0
    batch_size: int = 5


class ModelPaperGenerator:
    """
    Generator for O/L Mathematics model paper questions.
    Separate methods for each question type with specialized prompts.
    """
    
    def __init__(self, api_key: str):
        """Initialize the generator"""
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        self.model_name = "gemini-2.5-flash"
        self.model = None
        
        self.last_request_time = 0
        self.min_request_interval = 2
        
        self.generation_config = {
            'temperature': 0.8,
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 8192,
        }
        
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        # Past paper data
        self.past_paper_questions: List[Dict] = []
        self.past_paper_by_topic: Dict[str, List[Dict]] = {}
        self.past_paper_by_type: Dict[str, List[Dict]] = {}
        self.available_topics: List[str] = []
        self.past_papers_loaded = False
        
        print("✅ Model Paper Generator initialized")
    
    def _ensure_model(self):
        """Lazy load the Gemini model"""
        if self.model is None:
            print(f"Loading model: {self.model_name}")
            self.model = genai.GenerativeModel(self.model_name)
    
    def _rate_limit_wait(self):
        """Implement rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            wait_time = self.min_request_interval - elapsed
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    def _parse_topics(self, topic_string: str) -> List[str]:
        """Parse topic string - handles combined topics with '/'"""
        if not topic_string:
            return []
        topics = [t.strip() for t in topic_string.split('/')]
        return [t for t in topics if t]
    
    # ==================== Data Loading ====================
    
    def load_past_paper_questions(self, json_path: str) -> bool:
        """Load past paper questions from JSON file."""
        print(f"\n{'='*60}")
        print("📚 LOADING PAST PAPER QUESTIONS")
        print(f"{'='*60}")
        
        # Try multiple paths
        possible_paths = [
            json_path,
            os.path.join(os.path.dirname(__file__), '..', '..', json_path),
            os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'extracted_text', 'model_paper_questions.json'),
            os.path.abspath(json_path),
        ]
        
        actual_path = None
        for path in possible_paths:
            normalized_path = os.path.normpath(path)
            if os.path.exists(normalized_path):
                actual_path = normalized_path
                print(f"✅ Found file at: {actual_path}")
                break
        
        if actual_path is None:
            print(f"❌ File not found")
            self.past_papers_loaded = False
            return False
        
        try:
            with open(actual_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            questions = data.get('questions', [])
            
            if not questions:
                print("❌ No questions found in the file")
                self.past_papers_loaded = False
                return False
            
            self.past_paper_questions = []
            self.past_paper_by_topic = {}
            self.past_paper_by_type = {}
            all_topics = set()
            
            for q in questions:
                self.past_paper_questions.append(q)
                
                topic_str = q.get('topic', '')
                topics = self._parse_topics(topic_str)
                
                for topic in topics:
                    all_topics.add(topic)
                    if topic not in self.past_paper_by_topic:
                        self.past_paper_by_topic[topic] = []
                    self.past_paper_by_topic[topic].append(q)
                
                q_type = q.get('type', 'short_answer')
                if q_type not in self.past_paper_by_type:
                    self.past_paper_by_type[q_type] = []
                self.past_paper_by_type[q_type].append(q)
            
            self.available_topics = list(all_topics)
            self.past_papers_loaded = True
            
            print(f"📊 Loaded {len(self.past_paper_questions)} questions")
            print(f"📚 Topics: {len(self.available_topics)}")
            print(f"📝 Types: {list(self.past_paper_by_type.keys())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading past papers: {e}")
            self.past_papers_loaded = False
            return False
    
    def get_statistics(self) -> Dict:
        """Get statistics about loaded data."""
        return {
            "past_papers_loaded": self.past_papers_loaded,
            "total_questions": len(self.past_paper_questions),
            "available_topics": self.available_topics,
            "questions_by_type": {k: len(v) for k, v in self.past_paper_by_type.items()},
            "questions_by_topic": {k: len(v) for k, v in self.past_paper_by_topic.items()}
        }
    
    def _get_reference_questions(self, topics: List[str], question_type: str, count: int = 2) -> List[Dict]:
        """Get reference questions from past papers."""
        candidates = []
        
        for topic in topics:
            topic_questions = self.past_paper_by_topic.get(topic, [])
            for q in topic_questions:
                if q.get('type') == question_type and q not in candidates:
                    candidates.append(q)
        
        if len(candidates) < count:
            type_questions = self.past_paper_by_type.get(question_type, [])
            for q in type_questions:
                if q not in candidates:
                    candidates.append(q)
                if len(candidates) >= count * 2:
                    break
        
        return random.sample(candidates, min(count, len(candidates))) if candidates else []
    
    def _select_topics(self, count: int) -> List[str]:
        """Select random topics for questions."""
        if not self.available_topics:
            return ["ගණිතය"] * count
        
        topics = []
        available = self.available_topics.copy()
        random.shuffle(available)
        
        for i in range(count):
            if not available:
                available = self.available_topics.copy()
                random.shuffle(available)
            topics.append(available.pop(0))
        
        return topics
    
    # ==================== SHORT ANSWER GENERATION ====================
    
    def _build_short_answer_prompt(self, topics: List[str], count: int, references: List[Dict]) -> str:
        """Build prompt for short answer questions."""
        
        # Format reference examples
        ref_text = ""
        if references:
            ref_text = "\n=== ආදර්ශ උදාහරණ ===\n"
            for i, ref in enumerate(references[:2], 1):
                ref_text += f"\nඋදාහරණ {i}:\n"
                ref_text += f"මාතෘකාව: {ref.get('topic', '')}\n"
                ref_text += f"ප්‍රශ්නය: {ref.get('question', '')}\n"
                
                final_ans = ref.get('final_answer', [])
                if final_ans:
                    ref_text += "පිළිතුරු පියවර:\n"
                    for step in final_ans[:4]:
                        if isinstance(step, dict):
                            desc = step.get('step', '') or ''
                            val = step.get('answer', '')
                            if val:
                                ref_text += f"  • {desc} = {val}\n"
        
        topics_str = ", ".join(topics[:5])
        
        prompt = f"""ඔබ O/L ගණිතය විභාග ප්‍රශ්න සාදන විශේෂඥ ගුරුවරයෙක්.

කාර්යය: කෙටි පිළිතුරු ප්‍රශ්න {count}ක් සාදන්න
මාතෘකා: {topics_str}

{ref_text}

=== නිමැවුම් ආකෘතිය ===

සෑම ප්‍රශ්නයක් සඳහා මෙම ආකෘතිය හරියටම අනුගමනය කරන්න:

QUESTION_START
NUMBER: [අංකය]
TOPIC: [මාතෘකාව]
QUESTION: [සම්පූර්ණ ගණිත ප්‍රශ්නය සිංහලෙන්]
STEPS:
- [පියවර 1 විස්තරය] = [ගණනය/පිළිතුර]
- [පියවර 2 විස්තරය] = [ගණනය/පිළිතුර]
- [පියවර 3 විස්තරය] = [ගණනය/පිළිතුර]
FINAL_ANSWER: [අවසාන පිළිතුර]
QUESTION_END

---

=== නීති ===
1. සෑම ප්‍රශ්නයකම STEPS අවම වශයෙන් 2-4ක් තිබිය යුතුය
2. "රු." මුදල් සඳහා භාවිතා කරන්න
3. ගණිත සංකේත: ², ³, π, √, ×, ÷
4. සෑම ප්‍රශ්නයකටම වෙනස් සංඛ්‍යා භාවිතා කරන්න
5. ප්‍රශ්න --- මගින් වෙන් කරන්න

=== ප්‍රශ්න වර්ග උදාහරණ ===
- සුළු කරන්න: (2/3x) + (5/6x) - (7/12x)
- සාධක සොයන්න: 2x² - 18
- සමීකරණය විසඳන්න: 3x + 5 = 20
- පොලිය ගණනය කරන්න: රු. 50000 ක් 8% පොලියට වසර 2ක්
- සම්භාවිතාව සොයන්න: දාදු කැටයක් දෙවරක් දැමූ විට...

දැන් ප්‍රශ්න {count}ක් සාදන්න:
"""
        return prompt
    
    def _parse_short_answer_response(self, text: str) -> List[Dict]:
        """Parse short answer questions from response."""
        questions = []
        
        # Split by QUESTION_START...QUESTION_END or ---
        pattern = r'QUESTION_START(.*?)QUESTION_END'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if not matches:
            # Try splitting by ---
            parts = text.split('---')
            for part in parts:
                if 'QUESTION:' in part or 'NUMBER:' in part:
                    matches.append(part)
        
        for match in matches:
            try:
                q_data = {}
                
                # Extract number
                num_match = re.search(r'NUMBER:\s*(\d+)', match)
                q_data['question_number'] = int(num_match.group(1)) if num_match else len(questions) + 1
                
                # Extract topic
                topic_match = re.search(r'TOPIC:\s*(.+?)(?:\n|$)', match)
                q_data['topics'] = [topic_match.group(1).strip()] if topic_match else []
                
                # Extract question
                q_match = re.search(r'QUESTION:\s*(.+?)(?=\nSTEPS:|$)', match, re.DOTALL)
                q_data['question'] = q_match.group(1).strip() if q_match else ""
                
                # Extract steps
                steps = []
                steps_match = re.search(r'STEPS:(.*?)(?=FINAL_ANSWER:|$)', match, re.DOTALL)
                if steps_match:
                    step_lines = steps_match.group(1).strip().split('\n')
                    for line in step_lines:
                        line = line.strip().lstrip('-').strip()
                        if '=' in line:
                            parts = line.split('=', 1)
                            steps.append({
                                "description": parts[0].strip(),
                                "value": parts[1].strip()
                            })
                        elif line:
                            steps.append({
                                "description": line,
                                "value": ""
                            })
                q_data['answer_steps'] = steps
                
                # Extract final answer
                final_match = re.search(r'FINAL_ANSWER:\s*(.+?)(?:\n|$)', match)
                q_data['final_answer'] = final_match.group(1).strip() if final_match else ""
                
                if q_data['question'] and len(q_data['question']) > 10:
                    questions.append(q_data)
                    
            except Exception as e:
                print(f"  ⚠️ Parse error: {e}")
                continue
        
        return questions
    
    def generate_short_answer_questions(
        self,
        count: int = 5,
        topics: Optional[List[str]] = None,
        api_delay: float = 4.0
    ) -> Dict:
        """
        Generate short answer questions.
        
        Args:
            count: Number of questions to generate (1-10)
            topics: Optional list of topics to use
            api_delay: Delay between API calls
        
        Returns:
            Dict with questions and metadata
        """
        print(f"\n{'='*60}")
        print(f"📝 GENERATING {count} SHORT ANSWER QUESTIONS")
        print(f"{'='*60}")
        
        if not self.past_papers_loaded:
            raise ValueError("Past papers not loaded. Call load_past_paper_questions() first.")
        
        self._ensure_model()
        start_time = time.time()
        
        # Select topics if not provided
        if not topics:
            topics = self._select_topics(count)
        
        print(f"📚 Topics: {topics[:5]}...")
        
        # Get reference questions
        references = self._get_reference_questions(topics, 'short_answer', count=2)
        
        all_questions = []
        batch_size = min(5, count)
        attempts = 0
        max_attempts = 3
        
        while len(all_questions) < count and attempts < max_attempts:
            attempts += 1
            remaining = count - len(all_questions)
            batch_count = min(batch_size, remaining + 2)
            
            print(f"\n  Attempt {attempts}: Generating {batch_count} questions...")
            
            self._rate_limit_wait()
            
            try:
                prompt = self._build_short_answer_prompt(topics, batch_count, references)
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                
                if response.text:
                    new_questions = self._parse_short_answer_response(response.text)
                    
                    for q in new_questions:
                        if len(all_questions) < count:
                            q['question_number'] = len(all_questions) + 1
                            all_questions.append(q)
                    
                    print(f"  ✅ Parsed {len(new_questions)} questions. Total: {len(all_questions)}/{count}")
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:100]}")
                time.sleep(api_delay)
        
        generation_time = round(time.time() - start_time, 2)
        
        return {
            "type": "short_answer",
            "questions": all_questions,
            "count": len(all_questions),
            "requested": count,
            "topics_used": list(set(topics)),
            "generation_time_seconds": generation_time
        }
    
    # ==================== STRUCTURED QUESTION GENERATION ====================
    
    def _build_structured_prompt(self, topics: List[str], count: int, references: List[Dict]) -> str:
        """Build prompt for structured questions with sub-questions."""
        
        # Format reference examples
        ref_text = ""
        if references:
            ref_text = "\n=== ආදර්ශ ව්‍යුහගත ප්‍රශ්න ===\n"
            for i, ref in enumerate(references[:2], 1):
                ref_text += f"\nඋදාහරණ {i}:\n"
                ref_text += f"ප්‍රධාන ප්‍රශ්නය: {ref.get('question', '')[:200]}...\n"
                
                sub_qs = ref.get('sub_questions', [])
                if sub_qs:
                    ref_text += "උප ප්‍රශ්න:\n"
                    for j, sq in enumerate(sub_qs[:3]):
                        sq_text = sq.get('sub_question', '')[:100]
                        ref_text += f"  ({chr(ord('අ') + j)}) {sq_text}...\n"
        
        topics_str = ", ".join(topics)
        
        prompt = f"""ඔබ O/L ගණිතය විභාග ප්‍රශ්න සාදන විශේෂඥ ගුරුවරයෙක්.

කාර්යය: ව්‍යුහගත (Structured) ප්‍රශ්න {count}ක් සාදන්න
මාතෘකා: {topics_str}

{ref_text}

=== ව්‍යුහගත ප්‍රශ්නයක ලක්ෂණ ===
1. ප්‍රධාන සන්දර්භයක් හෝ තත්ත්වයක් විස්තර කරයි
2. උප ප්‍රශ්න 3-5ක් අඩංගු වේ (අ, ආ, ඇ, ඈ, ඉ)
3. උප ප්‍රශ්න එකිනෙකට සම්බන්ධ වේ
4. සෑම උප ප්‍රශ්නයකම පිළිතුරු පියවර 1-3ක් ඇත

=== නිමැවුම් ආකෘතිය ===

STRUCTURED_START
NUMBER: [අංකය]
TOPIC: [මාතෘකාව]
MAIN_CONTEXT: [ප්‍රධාන සන්දර්භය - සිද්ධිය විස්තර කරන්න, උදා: "රවී රුපියල් 80000ක් බැංකුවක 12% වාර්ෂික පොලී අනුපාතයකට තැන්පත් කරයි."]

SUB_QUESTION: (අ)
TEXT: [පළමු උප ප්‍රශ්නය - ප්‍රශ්නයක් ලෙස ලියන්න, උදා: "පළමු වසර අවසානයේ ලැබෙන පොලිය කීයද?"]
STEPS:
- [පියවර විස්තරය] = [ගණනය/පිළිතුර]
- [පියවර විස්තරය] = [ගණනය/පිළිතුර]
ANSWER: [මෙම උප ප්‍රශ්නයේ පිළිතුර]

SUB_QUESTION: (ආ)
TEXT: [දෙවන උප ප්‍රශ්නය]
STEPS:
- [පියවර] = [පිළිතුර]
ANSWER: [පිළිතුර]

SUB_QUESTION: (ඇ)
TEXT: [තෙවන උප ප්‍රශ්නය]
STEPS:
- [පියවර] = [පිළිතුර]
ANSWER: [පිළිතුර]

SUB_QUESTION: (ඈ)
TEXT: [සිව්වන උප ප්‍රශ්නය]
STEPS:
- [පියවර] = [පිළිතුර]
ANSWER: [පිළිතුර]

STRUCTURED_END

---

=== වැදගත් නීති ===
1. MAIN_CONTEXT යනු ප්‍රශ්නයක් නොවේ - එය සිද්ධියක් හෝ තත්ත්වයක් විස්තර කිරීමකි
2. සෑම SUB_QUESTION එකක්ම ප්‍රශ්නයක් විය යුතුය (? සලකුණ භාවිතා කරන්න)
3. උප ප්‍රශ්න අවම වශයෙන් 3ක් සහ උපරිම 5ක් තිබිය යුතුය
4. සෑම උප ප්‍රශ්නයකටම STEPS සහ ANSWER තිබිය යුතුය
5. උප ප්‍රශ්න එකිනෙකට සම්බන්ධ විය යුතුය (පෙර පිළිතුරු පසු ප්‍රශ්නවලට අවශ්‍ය විය හැක)

=== ප්‍රශ්න සන්දර්භ උදාහරණ ===
- පොලිය: "සුමන රුපියල් 50000ක් බැංකුවක 10% වාර්ෂික පොලී අනුපාතයකට තැන්පත් කරයි..."
- කොටස්: "සමාගමක කොටස් 10000ක් නිකුත් කර ඇත. කොටසක මිල රුපියල් 25 කි..."
- බදු: "ජයන්ත මසකට රුපියල් 120000ක වැටුපක් ලබයි. ආදායම් බදු අනුපාතය 6% කි..."

දැන් ව්‍යුහගත ප්‍රශ්න {count}ක් සාදන්න:
"""
        return prompt
    
    def _parse_structured_response(self, text: str) -> List[Dict]:
        """Parse structured questions from response."""
        questions = []
        
        # Split by STRUCTURED_START...STRUCTURED_END
        pattern = r'STRUCTURED_START(.*?)STRUCTURED_END'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if not matches:
            # Try splitting by ---
            parts = text.split('---')
            for part in parts:
                if 'MAIN_CONTEXT:' in part or 'SUB_QUESTION:' in part:
                    matches.append(part)
        
        for match in matches:
            try:
                q_data = {}
                
                # Extract number
                num_match = re.search(r'NUMBER:\s*(\d+)', match)
                q_data['question_number'] = int(num_match.group(1)) if num_match else len(questions) + 1
                
                # Extract topic
                topic_match = re.search(r'TOPIC:\s*(.+?)(?:\n|$)', match)
                q_data['topics'] = [topic_match.group(1).strip()] if topic_match else []
                
                # Extract main context
                context_match = re.search(r'MAIN_CONTEXT:\s*(.+?)(?=\nSUB_QUESTION:|$)', match, re.DOTALL)
                q_data['question'] = context_match.group(1).strip() if context_match else ""
                
                # Extract sub-questions
                sub_questions = []
                sub_pattern = r'SUB_QUESTION:\s*\(([අ-ඉa-e\d]+)\)\s*\nTEXT:\s*(.+?)(?=\nSTEPS:|$)(.*?)(?=\nSUB_QUESTION:|STRUCTURED_END|$)'
                sub_matches = re.findall(sub_pattern, match, re.DOTALL)
                
                # If pattern doesn't match, try alternative
                if not sub_matches:
                    sub_pattern2 = r'SUB_QUESTION:\s*\(([^)]+)\)[^\n]*\n(?:TEXT:\s*)?(.+?)(?:\n\s*STEPS:|ANSWER:)(.*?)(?=SUB_QUESTION:|STRUCTURED_END|---|\Z)'
                    sub_matches = re.findall(sub_pattern2, match, re.DOTALL)
                
                for label, sub_text, rest in sub_matches:
                    sub_q = {
                        "sub_question_label": f"({label.strip()})",
                        "sub_question": sub_text.strip(),
                        "answer_steps": []
                    }
                    
                    # Extract steps
                    steps_match = re.search(r'STEPS:(.*?)(?=ANSWER:|SUB_QUESTION:|$)', rest, re.DOTALL)
                    if steps_match:
                        step_lines = steps_match.group(1).strip().split('\n')
                        for line in step_lines:
                            line = line.strip().lstrip('-').strip()
                            if '=' in line:
                                parts = line.split('=', 1)
                                sub_q['answer_steps'].append({
                                    "description": parts[0].strip(),
                                    "value": parts[1].strip()
                                })
                    
                    # Extract answer
                    ans_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', rest)
                    if ans_match:
                        sub_q['answer'] = ans_match.group(1).strip()
                    
                    if sub_q['sub_question']:
                        sub_questions.append(sub_q)
                
                q_data['sub_questions'] = sub_questions
                
                # Only add if we have main question and at least 2 sub-questions
                if q_data['question'] and len(sub_questions) >= 2:
                    questions.append(q_data)
                else:
                    print(f"  ⚠️ Skipped: main_q={bool(q_data['question'])}, sub_qs={len(sub_questions)}")
                    
            except Exception as e:
                print(f"  ⚠️ Parse error: {e}")
                continue
        
        return questions
    
    def generate_structured_questions(
        self,
        count: int = 3,
        topics: Optional[List[str]] = None,
        api_delay: float = 4.0
    ) -> Dict:
        """
        Generate structured questions with sub-questions.
        
        Args:
            count: Number of questions to generate (1-5)
            topics: Optional list of topics to use
            api_delay: Delay between API calls
        
        Returns:
            Dict with questions and metadata
        """
        print(f"\n{'='*60}")
        print(f"📋 GENERATING {count} STRUCTURED QUESTIONS")
        print(f"{'='*60}")
        
        if not self.past_papers_loaded:
            raise ValueError("Past papers not loaded. Call load_past_paper_questions() first.")
        
        self._ensure_model()
        start_time = time.time()
        
        # Select topics if not provided
        if not topics:
            topics = self._select_topics(count)
        
        print(f"📚 Topics: {topics}")
        
        # Get reference questions
        references = self._get_reference_questions(topics, 'structured', count=2)
        
        all_questions = []
        attempts = 0
        max_attempts = 4
        
        # Generate one at a time for better quality
        while len(all_questions) < count and attempts < max_attempts:
            attempts += 1
            remaining = count - len(all_questions)
            
            print(f"\n  Attempt {attempts}: Generating {min(2, remaining)} structured questions...")
            
            self._rate_limit_wait()
            
            try:
                prompt = self._build_structured_prompt(topics, min(2, remaining), references)
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                
                if response.text:
                    new_questions = self._parse_structured_response(response.text)
                    
                    for q in new_questions:
                        if len(all_questions) < count:
                            q['question_number'] = len(all_questions) + 1
                            all_questions.append(q)
                    
                    print(f"  ✅ Parsed {len(new_questions)} questions. Total: {len(all_questions)}/{count}")
                
                time.sleep(api_delay)
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:100]}")
                time.sleep(api_delay)
        
        generation_time = round(time.time() - start_time, 2)
        
        return {
            "type": "structured",
            "questions": all_questions,
            "count": len(all_questions),
            "requested": count,
            "topics_used": list(set(topics)),
            "generation_time_seconds": generation_time
        }
    
    # ==================== ESSAY TYPE GENERATION ====================
    
    def _build_essay_prompt(self, topics: List[str], count: int, references: List[Dict]) -> str:
        """Build prompt for essay type questions with real-life scenarios."""
        
        # Format reference examples
        ref_text = ""
        if references:
            ref_text = "\n=== ආදර්ශ රචනා ප්‍රශ්න ===\n"
            for i, ref in enumerate(references[:2], 1):
                ref_text += f"\nඋදාහරණ {i}:\n"
                ref_text += f"ප්‍රශ්නය: {ref.get('question', '')[:300]}...\n"
        
        topics_str = ", ".join(topics)
        
        prompt = f"""ඔබ O/L ගණිතය විභාග ප්‍රශ්න සාදන විශේෂඥ ගුරුවරයෙක්.

කාර්යය: රචනා වර්ගයේ (Essay Type) ප්‍රශ්න {count}ක් සාදන්න
මාතෘකා: {topics_str}

{ref_text}

=== රචනා ප්‍රශ්නයක ලක්ෂණ ===
1. සැබෑ ජීවිත තත්ත්වයක් විස්තරාත්මකව ඉදිරිපත් කරයි
2. විස්තරය දිග විය යුතුය (වාක්‍ය 3-5)
3. උප ප්‍රශ්න 4-6ක් අඩංගු වේ - (i), (ii), (iii), (iv), (v)
4. උප ප්‍රශ්න එකිනෙකට සම්බන්ධ සහ ප්‍රගතිශීලී
5. අවසාන උප ප්‍රශ්නය සාමාන්‍යයෙන් සාරාංශයක් හෝ සංසන්දනයක්

=== සැබෑ ජීවිත සිද්ධි උදාහරණ ===
- "කමල් තම නිවස මසකට රුපියල් 8000 බැගින් වර්ෂයකට බදු දී එම මුදල් එකවර ලබාගනියි..."
- "එකක් රුපියල් 84000 බැගින් වටිනා රූපවාහිනී තොගයක් විකිණීමට තිබේ. රුවිනි..."
- "අමලා සහ සුමනා නිවාඩු කාලය තුළදී එක්තරා නවකතාවක් කියවීමට තීරණය කරති..."
- "සාදයකට සහභාගි වූ වැඩිහිටියන්ටත් ළමයින්ටත් රසකැවිලිවලින් සංග්‍රහ කිරීම සඳහා..."

=== නිමැවුම් ආකෘතිය ===

ESSAY_START
NUMBER: [අංකය]
TOPICS: [මාතෘකා කොමාවෙන් වෙන් කර]
SCENARIO: [සැබෑ ජීවිත සිද්ධිය විස්තරාත්මකව - අවම වශයෙන් වාක්‍ය 3ක්. පුද්ගලයන්ගේ නම්, මුදල් ප්‍රමාණ, ප්‍රතිශත, කාල සීමා ආදිය ඇතුළත් කරන්න.]

SUB_QUESTION: (i)
TEXT: [පළමු උප ප්‍රශ්නය - ? සලකුණ සමඟ]
STEPS:
- [පියවර] = [ගණනය]
- [පියවර] = [පිළිතුර]
ANSWER: [පිළිතුර]

SUB_QUESTION: (ii)
TEXT: [දෙවන උප ප්‍රශ්නය]
STEPS:
- [පියවර] = [ගණනය]
ANSWER: [පිළිතුර]

SUB_QUESTION: (iii)
TEXT: [තෙවන උප ප්‍රශ්නය]
STEPS:
- [පියවර] = [ගණනය]
ANSWER: [පිළිතුර]

SUB_QUESTION: (iv)
TEXT: [සිව්වන උප ප්‍රශ්නය]
STEPS:
- [පියවර] = [ගණනය]
ANSWER: [පිළිතුර]

SUB_QUESTION: (v)
TEXT: [පස්වන උප ප්‍රශ්නය - සංසන්දනයක් හෝ නිගමනයක්]
STEPS:
- [පියවර] = [ගණනය]
ANSWER: [පිළිතුර]

ESSAY_END

---

=== වැදගත් නීති ===
1. SCENARIO යනු සැබෑ ජීවිත සිද්ධියක් - විස්තරාත්මක විය යුතුය
2. පුද්ගලයන්ගේ සිංහල නම් භාවිතා කරන්න (සුමන, කමල්, නිමාලි, රවී, ආදිය)
3. සෑම SUB_QUESTION එකක්ම ප්‍රශ්නයක් විය යුතුය (? සලකුණ)
4. උප ප්‍රශ්න අවම 4ක් සහ උපරිම 6ක්
5. උප ප්‍රශ්න පෙර පිළිතුරු මත රඳා පවතිය හැක
6. අවසාන ප්‍රශ්නය සාමාන්‍යයෙන් "පෙන්වන්න", "සංසන්දනය කරන්න", "තීරණය කරන්න" වර්ගයේ

දැන් රචනා ප්‍රශ්න {count}ක් සාදන්න:
"""
        return prompt
    
    def _parse_essay_response(self, text: str) -> List[Dict]:
        """Parse essay type questions from response."""
        questions = []
        
        # Split by ESSAY_START...ESSAY_END
        pattern = r'ESSAY_START(.*?)ESSAY_END'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if not matches:
            # Try splitting by ---
            parts = text.split('---')
            for part in parts:
                if 'SCENARIO:' in part or 'SUB_QUESTION:' in part:
                    matches.append(part)
        
        for match in matches:
            try:
                q_data = {}
                
                # Extract number
                num_match = re.search(r'NUMBER:\s*(\d+)', match)
                q_data['question_number'] = int(num_match.group(1)) if num_match else len(questions) + 1
                
                # Extract topics
                topics_match = re.search(r'TOPICS?:\s*(.+?)(?:\n|$)', match)
                if topics_match:
                    topics_str = topics_match.group(1).strip()
                    q_data['topics'] = [t.strip() for t in topics_str.split(',')]
                else:
                    q_data['topics'] = []
                
                # Extract scenario
                scenario_match = re.search(r'SCENARIO:\s*(.+?)(?=\nSUB_QUESTION:|$)', match, re.DOTALL)
                q_data['question'] = scenario_match.group(1).strip() if scenario_match else ""
                
                # Extract sub-questions (same pattern as structured)
                sub_questions = []
                sub_pattern = r'SUB_QUESTION:\s*\(([ivxIVX\d]+)\)[^\n]*\n(?:TEXT:\s*)?(.+?)(?:\n\s*STEPS:|ANSWER:)(.*?)(?=SUB_QUESTION:|ESSAY_END|---|\Z)'
                sub_matches = re.findall(sub_pattern, match, re.DOTALL)
                
                for label, sub_text, rest in sub_matches:
                    sub_q = {
                        "sub_question_label": f"({label.strip()})",
                        "sub_question": sub_text.strip(),
                        "answer_steps": []
                    }
                    
                    # Extract steps
                    steps_match = re.search(r'STEPS:(.*?)(?=ANSWER:|SUB_QUESTION:|$)', rest, re.DOTALL)
                    if steps_match:
                        step_lines = steps_match.group(1).strip().split('\n')
                        for line in step_lines:
                            line = line.strip().lstrip('-').strip()
                            if '=' in line:
                                parts = line.split('=', 1)
                                sub_q['answer_steps'].append({
                                    "description": parts[0].strip(),
                                    "value": parts[1].strip()
                                })
                    
                    # Extract answer
                    ans_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', rest)
                    if ans_match:
                        sub_q['answer'] = ans_match.group(1).strip()
                    
                    if sub_q['sub_question']:
                        sub_questions.append(sub_q)
                
                q_data['sub_questions'] = sub_questions
                
                # Only add if we have scenario and at least 3 sub-questions
                if q_data['question'] and len(q_data['question']) > 50 and len(sub_questions) >= 3:
                    questions.append(q_data)
                else:
                    print(f"  ⚠️ Skipped: scenario_len={len(q_data.get('question', ''))}, sub_qs={len(sub_questions)}")
                    
            except Exception as e:
                print(f"  ⚠️ Parse error: {e}")
                continue
        
        return questions
    
    def generate_essay_questions(
        self,
        count: int = 5,
        topics: Optional[List[str]] = None,
        api_delay: float = 4.0
    ) -> Dict:
        """
        Generate essay type questions with real-life scenarios.
        
        Args:
            count: Number of questions to generate (1-5)
            topics: Optional list of topics to use
            api_delay: Delay between API calls
        
        Returns:
            Dict with questions and metadata
        """
        print(f"\n{'='*60}")
        print(f"📝 GENERATING {count} ESSAY TYPE QUESTIONS")
        print(f"{'='*60}")
        
        if not self.past_papers_loaded:
            raise ValueError("Past papers not loaded. Call load_past_paper_questions() first.")
        
        self._ensure_model()
        start_time = time.time()
        
        # Select topics if not provided
        if not topics:
            topics = self._select_topics(count * 2)  # More topics for essay
        
        print(f"📚 Topics: {topics}")
        
        # Get reference questions
        references = self._get_reference_questions(topics, 'essay_type', count=2)
        
        all_questions = []
        attempts = 0
        max_attempts = 4
        
        # Generate one at a time for best quality
        while len(all_questions) < count and attempts < max_attempts:
            attempts += 1
            
            print(f"\n  Attempt {attempts}: Generating essay question...")
            
            self._rate_limit_wait()
            
            try:
                prompt = self._build_essay_prompt(topics, 1, references)
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                
                if response.text:
                    new_questions = self._parse_essay_response(response.text)
                    
                    for q in new_questions:
                        if len(all_questions) < count:
                            q['question_number'] = len(all_questions) + 1
                            all_questions.append(q)
                    
                    print(f"  ✅ Parsed {len(new_questions)} questions. Total: {len(all_questions)}/{count}")
                
                time.sleep(api_delay)
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:100]}")
                time.sleep(api_delay)
        
        generation_time = round(time.time() - start_time, 2)
        
        return {
            "type": "essay_type",
            "questions": all_questions,
            "count": len(all_questions),
            "requested": count,
            "topics_used": list(set(topics)),
            "generation_time_seconds": generation_time
        }