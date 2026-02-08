import json
import os
import re
import time
from typing import List, Dict, Optional, Tuple

import google.generativeai as genai


class SinhalaRAGSystem: 
    """
    RAG System for Sinhala Mathematics Question Generation
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the RAG system
        
        Args: 
            api_key: Google Gemini API key
        """
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Model configuration
        self.model_name = "gemini-2.5-flash"
        self.model = None
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 2
        
        # Generation config
        self.generation_config = {
            'temperature': 0.8,
            'top_p':  0.95,
            'top_k': 40,
            'max_output_tokens':  16384,
        }
        
        # Safety settings
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        # ChromaDB components
        self.chroma_client = None
        self.embedding_fn = None
        self.collections = {}
        self.data = {}
        self.data_loaded = False
        
        # Initialize ChromaDB
        self._setup_chromadb()
        
        print(f" RAG System initialized with model: {self.model_name}")
    
    # ==================== Setup Methods ====================
    
    def _setup_chromadb(self):
        """Setup ChromaDB and embedding function"""
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            
            self.chroma_client = chromadb.Client()
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="paraphrase-multilingual-mpnet-base-v2"
            )
            print(" ChromaDB initialized with multilingual embeddings")
            
        except ImportError as e:
            print(f" ChromaDB not available: {e}")
            print("   Install with: pip install chromadb sentence-transformers")
        except Exception as e:
            print(f" ChromaDB setup error: {e}")
    
    def _ensure_model(self):
        """Lazy load the Gemini model"""
        if self.model is None:
            print(f"📦 Loading model: {self.model_name}")
            self.model = genai.GenerativeModel(self.model_name)
    
    def _rate_limit_wait(self):
        """Implement rate limiting for free tier"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            wait_time = self.min_request_interval - elapsed
            print(f"    Rate limit: waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    # ==================== Data Loading ====================
    
    def load_all_data(
        self,
        examples_path: str = "data/extracted_text/extracted_examples.json",
        exercises_path: str = "data/extracted_text/exteacted_exercises.json",
        paragraphs_path: str = "data/extracted_text/paragraphs_and_tables.json",
        guidelines_path:  str = "data/extracted_text/guidelines.json"
    ) -> bool:
        """
        Load all data files into ChromaDB
        
        Args:
            examples_path: Path to examples JSON file
            exercises_path: Path to exercises JSON file
            paragraphs_path: Path to paragraphs JSON file
            guidelines_path: Path to guidelines JSON file
            
        Returns:
            bool: True if at least one data source was loaded
        """
        if not self.chroma_client: 
            print(" ChromaDB not available, skipping data loading")
            return False
        
        print("\n" + "=" * 60)
        print("LOADING RAG DATA")
        print("=" * 60)
        
        # Setup collections
        self._setup_collections()
        
        # Load each data source
        paths = {
            'examples': examples_path,
            'exercises':  exercises_path,
            'paragraphs': paragraphs_path,
            'guidelines': guidelines_path
        }
        
        loaded_count = 0
        for name, path in paths.items():
            if os.path.exists(path):
                try:
                    self._load_data_file(name, path)
                    loaded_count += 1
                except Exception as e:
                    print(f" Error loading {name}: {e}")
            else:
                print(f" File not found: {path}")
        
        self.data_loaded = loaded_count > 0
        print(f"\n Data loading complete: {loaded_count} sources loaded")
        return self.data_loaded
    
    def _setup_collections(self):
        """Create ChromaDB collections"""
        collection_names = {
            'examples': 'sinhala_examples',
            'exercises': 'sinhala_exercises',
            'paragraphs': 'sinhala_paragraphs',
            'guidelines':  'sinhala_guidelines'
        }
        
        for key, name in collection_names.items():
            try:
                # Try to get existing collection
                self.collections[key] = self.chroma_client.get_collection(
                    name=name,
                    embedding_function=self.embedding_fn
                )
                print(f" Using existing collection: {name}")
            except Exception: 
                # Create new collection
                try:
                    self.collections[key] = self.chroma_client.create_collection(
                        name=name,
                        embedding_function=self.embedding_fn
                    )
                    print(f" Created collection: {name}")
                except Exception as e:
                    print(f" Failed to create {name}: {e}")
    
    def _load_data_file(self, name: str, path: str):
        """
        Load a specific data file into ChromaDB
        
        Args:
            name:  Collection name (examples, exercises, paragraphs, guidelines)
            path: File path to JSON data
        """
        print(f" Loading {name} from {path}...")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts, metadatas, ids = [], [], []
        
        if name == 'examples':
            examples = data.get('examples', [])
            for i, example in enumerate(examples):
                full_text = f"උදාහරණය:\n{example.get('Question', '')}\n\nවිසඳුම:\n"
                for step in example.get('Steps', []):
                    full_text += f"{step. get('Step', '')}\n"
                full_text += f"\nඅවසාන පිළිතුර:  {example.get('Final_answer', '')}"
                
                texts.append(full_text)
                metadatas.append({'type': 'example', 'index': i})
                ids.append(f"ex_{i}")
            self.data['examples'] = examples
            
        elif name == 'exercises': 
            exercises = data.get('exercises', [])
            for i, exercise in enumerate(exercises):
                main_q = exercise.get('metadata', {}).get('main_question', exercise.get('text', ''))
                full_text = f"අභ්‍යාස ප්‍රශ්නය:\n{main_q}"
                
                sub_qs = exercise.get('metadata', {}).get('sub_questions', [])
                if sub_qs:
                    full_text += "\n\nඅනු ප්‍රශ්න:\n"
                    for j, sub in enumerate(sub_qs, 1):
                        q_text = sub.get('question', sub.get('text', ''))
                        full_text += f"{j}. {q_text}\n"
                
                texts.append(full_text)
                metadatas.append({'type': 'exercise', 'index': i})
                ids.append(f"exr_{i}")
            self.data['exercises'] = exercises
            
        elif name == 'paragraphs':
            paragraphs = data.get('paragraphs', [])
            for i, para in enumerate(paragraphs):
                texts.append(para.get('text', ''))
                metadatas.append({'type': 'paragraph', 'page': para.get('page')})
                ids.append(para.get('id', f'para_{i}'))
            self.data['paragraphs'] = paragraphs
            
        elif name == 'guidelines':
            guidelines = data.get('guideline', [])
            for i, guideline in enumerate(guidelines):
                texts.append(guideline)
                metadatas.append({'type': 'guideline', 'index': i})
                ids.append(f"guide_{i}")
            self.data['guidelines'] = guidelines
        
        # Add to collection
        if texts and name in self.collections:
            try:
                self.collections[name].add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"    Loaded {len(texts)} {name}")
            except Exception as e: 
                print(f"    Error adding to collection: {e}")
    
    # ==================== Context Retrieval ====================
    
    def retrieve_context(self, query: str, n_results:  int = 3) -> Dict[str, List[Dict]]: 
        """
        Retrieve relevant context from all collections
        
        Args:
            query: Search query in Sinhala or English
            n_results: Number of results per collection
            
        Returns:
            Dictionary with collection names as keys and list of results as values
        """
        results = {}
        
        if not self.collections:
            return results
        
        for name, collection in self.collections.items():
            try:
                search = collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
                
                items = []
                if search.get('documents') and search['documents'][0]: 
                    for i in range(len(search['documents'][0])):
                        items.append({
                            'text': search['documents'][0][i],
                            'distance': search['distances'][0][i] if search. get('distances') else 0
                        })
                results[name] = items
                
            except Exception as e:
                print(f" Error querying {name}: {e}")
                results[name] = []
        
        return results
    
    # ==================== Prompt Building ====================
    
    def _build_prompt_with_context(
        self,
        topic: str,
        difficulty: str,
        num_questions: int,
        context:  Dict,
        existing_count: int = 0
    ) -> str:
        """
        Build generation prompt with RAG context
        
        Args: 
            topic: Mathematics topic in Sinhala
            difficulty:  easy, medium, or hard
            num_questions: Number of questions to generate
            context:  Retrieved context from ChromaDB
            existing_count:  Number of questions already generated (for continuation)
            
        Returns: 
            Complete prompt string
        """
        difficulty_config = {
            'easy': {
                'steps': '2-3',
                'description': 'simple calculations',
                'numbers': 'රු.  5,000 - රු. 50,000'
            },
            'medium': {
                'steps': '3-4',
                'description': 'multi-step problems',
                'numbers': 'රු. 50,000 - රු. 200,000'
            },
            'hard': {
                'steps': '4-5',
                'description': 'complex problems',
                'numbers': 'රු. 100,000 - රු. 500,000'
            }
        }
        
        config = difficulty_config.get(difficulty, difficulty_config['medium'])
        start_num = existing_count + 1
        
        # Build context section from RAG results
        context_section = ""
        
        if context. get('examples'):
            context_section += "\n REFERENCE EXAMPLES (use similar format):\n"
            for i, ex in enumerate(context['examples'][:2], 1):
                context_section += f"\nExample {i}:\n{ex['text'][: 500]}.. .\n"
        
        if context.get('guidelines'):
            context_section += "\n📋 GUIDELINES:\n"
            for guide in context['guidelines'][:2]: 
                context_section += f"- {guide['text'][:200]}\n"
        
        prompt = f"""You are an expert O/L mathematics teacher creating questions in Sinhala. 

{context_section}

TASK: Generate {num_questions} questions about "{topic}"
Difficulty: {difficulty} ({config['description']})
Number range: {config['numbers']}
Steps per solution: {config['steps']}

 IMPORTANT: Generate ALL {num_questions} complete questions.  Do NOT stop early. 

FORMAT for each question: 

QUESTION {start_num}:
[Complete math word problem in Sinhala]

SOLUTION:
පියවර 1: [Step description]
[Calculation] = [Result]

පියවර 2: [Step description]
[Calculation] = [Result]

පියවර 3: [Step description]
[Calculation] = [Result]

ANSWER:  [Final answer with රු. or %]

---

QUESTION {start_num + 1}:
[Different scenario with different numbers]

SOLUTION:
[Steps...]

ANSWER: [Answer]

---

RULES: 
✓ Generate EXACTLY {num_questions} complete questions
✓ Each question must have DIFFERENT numbers and scenarios
✓ Use Sinhala language
✓ Use "රු." for money, "%" for percentages
✓ Separate each question with ---
✓ Include SOLUTION and ANSWER for each

Generate {num_questions} questions about {topic}: 
"""
        return prompt
    
    # ==================== Response Parsing ====================
    
    def _parse_response(self, text: str) -> List[Dict]:
        """
        Parse generated questions from Gemini response
        
        Args:
            text: Raw response text from Gemini
            
        Returns:
            List of parsed question dictionaries
        """
        print(f"\n    Response:  {len(text)} chars")
        
        questions = []
        
        # Split by separator
        if '---' in text:
            parts = text.split('---')
            parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 50]
        else:
            parts = re.split(r'(? =QUESTION\s*\d+\s*: )', text, flags=re.IGNORECASE)
            parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 50]
        
        print(f"   📝 Found {len(parts)} sections")
        
        for part in parts:
            question_data = self._extract_question(part)
            if question_data:
                questions.append(question_data)
                print(f"    Question {len(questions)} parsed")
        
        return questions
    
    def _extract_question(self, section: str) -> Optional[Dict]:
        """
        Extract question, solution, and answer from a section
        
        Args: 
            section: Text section containing one question
            
        Returns: 
            Dictionary with question, solution, answer or None if parsing fails
        """
        q_text = None
        s_text = None
        a_text = "N/A"
        
        # Question patterns
        q_patterns = [
            r'QUESTION\s*\d*\s*:\s*(.+?)(?=\nSOLUTION|\nවිසඳුම|\n\n)',
            r'Question\s*\d*\s*:\s*(.+?)(?=\nSolution|\nවිසඳුම|\n\n)',
            r'ප්‍රශ්නය\s*\d*\s*:\s*(.+?)(?=\nවිසඳුම|\nSOLUTION|\n\n)',
        ]
        
        for pattern in q_patterns:
            match = re.search(pattern, section, re.DOTALL | re.IGNORECASE)
            if match:
                q_text = match.group(1).strip()
                if len(q_text) > 20:
                    break
        
        # Solution patterns
        s_patterns = [
            r'SOLUTION\s*:\s*(.+?)(?=\nANSWER|\nපිළිතුර|\nඅවසාන|$)',
            r'Solution\s*:\s*(.+?)(?=\nAnswer|\nපිළිතුර|\nඅවසාන|$)',
            r'විසඳුම\s*:\s*(.+?)(?=\nANSWER|\nපිළිතුර|\nඅවසාන|$)',
        ]
        
        for pattern in s_patterns:
            match = re.search(pattern, section, re.DOTALL | re.IGNORECASE)
            if match:
                s_text = match.group(1).strip()
                if len(s_text) > 20:
                    break
        
        # Answer patterns
        a_patterns = [
            r'ANSWER\s*:\s*(.+?)(?:\n|$)',
            r'Answer\s*:\s*(.+?)(?:\n|$)',
            r'පිළිතුර\s*:\s*(.+?)(?:\n|$)',
            r'අවසාන\s*පිළිතුර\s*:\s*(.+?)(?:\n|$)',
        ]
        
        for pattern in a_patterns:
            match = re.search(pattern, section, re.DOTALL | re.IGNORECASE)
            if match:
                a_text = match.group(1).strip().split('\n')[0].strip()
                if a_text:
                    break
        
        # Fallback:  get last calculation result as answer
        if a_text == "N/A" and s_text:
            lines = s_text.strip().split('\n')
            for line in reversed(lines):
                if '=' in line and ('රු' in line or '%' in line or any(c.isdigit() for c in line)):
                    a_text = line.split('=')[-1].strip()
                    break
        
        # Clean question text
        if q_text:
            q_text = re.sub(r'\s+', ' ', q_text).strip()
            q_text = re.sub(r'\s*(SOLUTION|විසඳුම)\s*: ?\s*$', '', q_text, flags=re.IGNORECASE).strip()
        
        # Validate
        if q_text and s_text and len(q_text) > 20 and len(s_text) > 20:
            return {
                'question': q_text,
                'solution': s_text,
                'answer':  a_text
            }
        
        return None
    
    # ==================== Main Generation ====================
    
    def generate_questions(
        self,
        topic: str,
        difficulty: str,
        num_questions: int
    ) -> Tuple[List[Dict], bool]:
        """
        Generate questions using RAG context
        
        Args:
            topic: Mathematics topic in Sinhala
            difficulty: easy, medium, or hard
            num_questions: Number of questions to generate
            
        Returns:
            Tuple of (list of questions, whether RAG context was used)
        """
        print(f"\n{'='*60}")
        print(f"GENERATING {num_questions} QUESTIONS WITH RAG")
        print(f"{'='*60}")
        print(f"Topic: {topic}")
        print(f"Difficulty:  {difficulty}")
        print(f"Model: {self.model_name}")
        print(f"RAG Data Loaded: {self.data_loaded}")
        
        self._ensure_model()
        
        # Retrieve context using RAG
        context = {}
        rag_used = False
        
        if self.data_loaded and self.collections:
            print("\n Retrieving RAG context...")
            context = self.retrieve_context(f"{topic} උදාහරණ ප්‍රශ්න", n_results=3)
            rag_used = any(len(items) > 0 for items in context.values())
            
            if rag_used:
                total_context = sum(len(items) for items in context.values())
                print(f"    Retrieved {total_context} context items")
            else:
                print("    No relevant context found")
        
        all_questions = []
        max_attempts = 5
        attempt = 0
        
        while len(all_questions) < num_questions and attempt < max_attempts: 
            attempt += 1
            remaining = num_questions - len(all_questions)
            
            print(f"\n Attempt {attempt}/{max_attempts} - Need {remaining} questions...")
            
            try:
                self._rate_limit_wait()
                
                request_count = min(remaining + 2, 7)
                
                prompt = self._build_prompt_with_context(
                    topic, difficulty, request_count, context, len(all_questions)
                )
                
                response = self. model.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                
                if not response.text:
                    print("    Empty response")
                    time.sleep(3)
                    continue
                
                new_questions = self._parse_response(response.text)
                
                if new_questions:
                    for q in new_questions:
                        if len(all_questions) < num_questions:
                            # Check duplicates
                            is_duplicate = any(
                                existing['question'][:50] == q['question'][:50]
                                for existing in all_questions
                            )
                            if not is_duplicate:
                                all_questions.append(q)
                    
                    print(f"   📊 Progress: {len(all_questions)}/{num_questions}")
                    
                    if len(all_questions) >= num_questions:
                        break
                else:
                    print("    No questions parsed")
                
                if len(all_questions) < num_questions:
                    time.sleep(2)
                    
            except Exception as e: 
                error_str = str(e).lower()
                print(f"    Error: {str(e)[:100]}")
                
                if "quota" in error_str or "rate" in error_str: 
                    wait_time = attempt * 10
                    print(f"    Rate limited.  Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    time.sleep(5)
        
        if all_questions:
            print(f"\n{'='*60}")
            print(f" Generated {len(all_questions)}/{num_questions} questions")
            print(f"   RAG Context Used: {rag_used}")
            print(f"{'='*60}")
            return all_questions[: num_questions], rag_used
        
        raise Exception("Failed to generate questions.  Please try again.")
    
    # ==================== Utility Methods ====================
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics about loaded collections"""
        stats = {}
        for name, collection in self. collections.items():
            try:
                stats[name] = collection.count()
            except: 
                stats[name] = 0
        return stats
    
    def export_questions(self, questions: List[Dict], path: str = "generated_questions.json"):
        """Export generated questions to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)
        print(f" Saved {len(questions)} questions to: {path}")


# ==================== Standalone Usage ====================

if __name__ == "__main__":
    """Test the RAG system standalone"""
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key: 
        print(" GEMINI_API_KEY not found in . env file")
        exit(1)
    
    print("\n" + "=" * 70)
    print("  SINHALA MATH RAG SYSTEM - STANDALONE TEST")
    print("=" * 70)
    
    # Initialize RAG system
    rag = SinhalaRAGSystem(api_key)
    
    # Load data
    rag.load_all_data()
    
    # Generate questions
    questions, rag_used = rag.generate_questions(
        topic="වාරික ගණනය",
        difficulty="medium",
        num_questions=3
    )
    
    # Display results
    print("\n" + "=" * 60)
    print(f"RESULTS: {len(questions)} QUESTIONS")
    print(f"RAG Context Used: {rag_used}")
    print("=" * 60)
    
    for i, q in enumerate(questions, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {q['question'][: 100]}...")
        print(f"A: {q['answer']}")
    
    # Export
    rag.export_questions(questions)