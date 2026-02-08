import json
import os
import re
import time
from typing import List, Dict, Optional, Tuple

import google.generativeai as genai


class SinhalaRAGSystem:
    """
    RAG System for Sinhala Mathematics Question Generation
    Supports multiple topics with topic-specific configurations
    """
    
    def __init__(self, api_key: str):
        """Initialize the RAG system"""
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
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 16384,
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
        
        # Topic-specific configurations
        self._setup_topic_configs()
        
        # Initialize ChromaDB
        self._setup_chromadb()
        
        print(f"RAG System initialized with model: {self.model_name}")
    
    # ==================== Topic Configuration ====================
    
    def _setup_topic_configs(self):
        """Setup topic-specific configurations"""
        self.topic_configs = {
            'පොළිය': {
                'difficulty': {
                    'easy': {
                        'steps': '2-3',
                        'description': 'simple interest calculations',
                        'numbers': 'රු. 5,000 - රු. 50,000',
                        'context': 'මූලික පොලී ගණනය කිරීම්'
                    },
                    'medium': {
                        'steps': '3-4',
                        'description': 'installment and reducing balance calculations',
                        'numbers': 'රු. 50,000 - රු. 200,000',
                        'context': 'වාරික ගණනය සහ හීන වන ශේෂය'
                    },
                    'hard': {
                        'steps': '4-5',
                        'description': 'compound interest and complex scenarios',
                        'numbers': 'රු. 100,000 - රු. 500,000',
                        'context': 'වැල් පොලිය සහ සංකීර්ණ ගණනය කිරීම්'
                    }
                },
                'prompt_template': """ප්‍රශ්නයේ සන්දර්භය:
- බැංකු ණය, තැන්පතු හෝ වාරික ගෙවීම් ගැන විය යුතුය
- පොලී අනුපාතික භාවිතා කරන්න (%)
- "රු." සංකේතය භාවිතා කරන්න
- ප්‍රායෝගික සන්දර්භයන් භාවිතා කරන්න (ගෘහ භාණ්ඩ, වාහන, ණය ආදිය)"""
            },
            
            'සමීකරණ': {
    'difficulty': {
        'easy': {
            'steps': '2-4',
            'description': 'simple linear equations with one variable',
            'numbers': '1-50',
            'context': 'සරල රේඛීය සමීකරණ',
            'sub_topics': ['සරල සමීකරණ'],
            'examples': [
                'භාග රහිත සරල සමීකරණ (2x + 8 = x + 12)',
                'සරල භාගමය සමීකරණ (x/2 + 1 = 3)',
                'එක් විචල්‍යයක් සහිත දෛනික ගැටළු'
            ]
        },
        'medium': {
            'steps': '4-8',
            'description': 'simultaneous equations and fractional equations',
            'numbers': '1-100 හෝ රු. 10,000 - රු. 100,000',
            'context': 'සමගාමී සමීකරණ සහ භාගමය සංගුණක',
            'sub_topics': ['සමගාමී සමීකරණ විසඳීම', 'භාගමය සංගුණක සහිත සමගාමී සමීකරණ'],
            'examples': [
                'දෙ විචල්‍යයන් සහිත සමගාමී සමීකරණ (6x + 2y = 1, 4x - y = 3)',
                'භාගමය සංගුණක සහිත සමීකරණ ((1/2)m + (2/3)n = 1)',
                'මුදල් බෙදාහැරීම් ගැටළු (කාසි, මුදල් ප්‍රමාණ)',
                'පාසල් උත්සව වැය ගණනය කිරීම්'
            ]
        },
        'hard': {
            'steps': '6-15',
            'description': 'quadratic equations and complex word problems',
            'numbers': 'විචල්‍ය සංඛ්‍යා හෝ දශම අගයන්',
            'context': 'වර්ගජ සමීකරණ සහ සංකීර්ණ ගැටළු',
            'sub_topics': [
                'සාධක භාවිතයෙන් වර්ගජ සමීකරණ විසඳීම',
                'වර්ග පූර්ණයෙන් වර්ගජ සමීකරණ විසදිම',
                'සූත්‍රය භාවිතයෙන් වර්ගජ සමීකරණ විසදීම'
            ],
            'examples': [
                'සාධකකරණය භාවිතයෙන් (x² - 5x + 6 = 0)',
                'වර්ග පූර්ණයෙන් (x² + 2x - 3 = 0)',
                'සූත්‍රය භාවිතයෙන් (2x² + 7x + 3 = 0)',
                'භාග සහිත වර්ගජ සමීකරණ ((3/(2x-1)) - (2/(3x+2)) = 1)',
                'ඍජුකෝණාස්‍රාකාර හැඩ ගැටළු',
                'පිතගෝරස් ප්‍රමේයය භාවිතා කරන ගැටළු',
                'සමාන්තර ශ්‍රේඪි ගැටළු'
            ],
            'formulas': [
                'x = (-b ± √(b² - 4ac)) / 2a',
                'පිතගෝරස් ප්‍රමේයය: a² + b² = c²'
            ]

            
        }
    },
    'prompt_template': """ප්‍රශ්නයේ සන්දර්භය:
- දෛනික ජීවිතයේ ගැටළු සමීකරණ භාවිතයෙන් විසඳන්න
- විචල්‍යයන් x, y භාවිතා කරන්න
- පියවරෙන් පියවර විසඳුම පෙන්වන්න
- සෑම පියවරක්ම සිංහලෙන් පැහැදිලි කරන්න

EASY සඳහා:
- සරල රේඛීය සමීකරණ භාවිතා කරන්න
- එක් විචල්‍යයක් පමණක් අඩංගු වන්න
- සරල වචන ගැටළු (අඹ බෙදීම, කාසි ගණන් කිරීම)

MEDIUM සඳහා:
- සමගාමී සමීකරණ යුගල භාවිතා කරන්න
- භාගමය සංගුණක ((1/2), (1/3)) ඇතුළත් කරන්න
- මුදල් බෙදාහැරීම්, පාසල් උත්සව ගැටළු
- රුපියල් හා සංඛ්‍යා මිශ්‍රණය කරන්න

HARD සඳහා:
- වර්ගජ සමීකරණ භාවිතා කරන්න (x² අඩංගු)
- තුන් ක්‍රමයෙන් එකක් භාවිතා කරන්න: සාධකකරණය, වර්ග පූර්ණය, හෝ සූත්‍රය
- ඍජුකෝණාස්‍රාකාර හැඩ, ත්‍රිකෝණ, ඉඩම් ප්‍රමාණ ගැටළු
- දශම පිළිතුරු ඇතුළත් විය හැක
- √ සංකේත භාවිතා කරන්න

අවසාන පිළිතුර:
- සරල සමීකරණ: x = 12 වැනි ආකාරයෙන්
- සමගාමී සමීකරණ: x = 20, y = 30 වැනි ආකාරයෙන්
- වර්ගජ සමීකරණ: x = 2 හෝ x = 3 වැනි ආකාරයෙන් (මූල දෙක)
- වචන ගැටළු: සන්දර්භයට අදාළව (දරුවන් ගණන = 12, ආදිය)"""
},
        
            'කොටස් වෙළෙඳපොළ': {
            'difficulty': {
                'easy': {
                    'steps': '2-4',
                    'description': 'basic share ownership and simple dividend calculations',
                    'numbers': 'කොටස් 100 - 10,000 | රු. 10 - රු. 100',
                    'context': 'කොටස් හිමිකාරිත්වය, භාග හා ප්‍රතිශත',
                    'sub_topics': [
                        'කොටස් හා හිමිකාරිත්වය',
                        'භාග ලෙස හිමිකාරිත්වය',
                        'ප්‍රතිශත ලෙස හිමිකාරිත්වය'
                    ],
                    'examples': [
                        'මුළු කොටස් අතරින් මිල දී ගත් කොටස් භාගයක් හා ප්‍රතිශතයක් ලෙස දක්වීම',
                        'කොටස් මිල × කොටස් ගණන = ආයෝජිත මුදල',
                        'සරල වාර්ෂික ලාභාංශ ගණනය'
                    ]
                },

                'medium': {
                    'steps': '4-8',
                    'description': 'dividend income and capital gain calculations',
                    'numbers': 'රු. 20,000 - රු. 100,000',
                    'context': 'ලාභාංශ, ප්‍රාග්ධන ලාභ, ප්‍රතිශත ආදායම',
                    'sub_topics': [
                        'ලාභාංශ ආදායම',
                        'වෙළෙඳපොළ මිල හා හඳුන්වා දීමේ මිල',
                        'ප්‍රාග්ධන ලාභය හා අලාභය'
                    ],
                    'examples': [
                        'ලාභාංශ = කොටස් ගණන × කොටසකට ලාභාංශ',
                        'විකුණුම් මිල − ගැණුම් මිල = ප්‍රාග්ධන ලාභය',
                        'ලාභය යෙදූ මුදලේ ප්‍රතිශතයක් ලෙස'
                    ]
                },

                'hard': {
                    'steps': '8-15',
                    'description': 'multiple investments, equations and comparative reasoning',
                    'numbers': 'රු. 50,000 - රු. 200,000',
                    'context': 'සමාගම් දෙකක් හෝ වැඩි ගණනක්, සමීකරණ භාවිතය',
                    'sub_topics': [
                        'සමගාමී ආයෝජන',
                        'x භාවිතයෙන් සමීකරණ ගොඩනගා විසඳීම',
                        'ලාභාංශ + ප්‍රාග්ධන ලාභ සංයෝජනය',
                        'අපේක්ෂිත ලාභ ප්‍රතිශතය පරීක්ෂා කිරීම'
                    ],
                    'examples': [
                        'A හා B සමාගම් දෙකක ආයෝජන සංසන්දනය',
                        'ලාභාංශ වෙනසක් මත සමීකරණයක් ගොඩනගා විසඳීම',
                        'අවසන් ලාභය යෙදූ මුදලේ ප්‍රතිශතයක් ලෙස විශ්ලේෂණය'
                    ],
                    'formulas': [
                        'ලාභාංශ ආදායම = කොටස් ගණන × කොටසකට ලාභාංශ',
                        'ප්‍රාග්ධන ලාභය = විකුණුම් මිල − ගැණුම් මිල',
                        'ප්‍රතිශත ලාභය = (ලාභය / යෙදූ මුදල) × 100'
                    ]
                }
            },

            'prompt_template': """ප්‍රශ්නයේ සන්දර්භය:
        - ලැයිස්තුගත සමාගමක් හා කොටස් වෙළෙඳපොළ සම්බන්ධ විය යුතුය
        - "රු." සංකේතය භාවිතා කරන්න
        - වාර්ෂික ලාභාංශ (per share) අනිවාර්යයෙන් සඳහන් කරන්න
        - වෙළෙඳපොළ මිල / හඳුන්වා දීමේ මිල පැහැදිලි කරන්න

        EASY සඳහා:
        - කොටස් හිමිකාරිත්වය භාගයක් හා ප්‍රතිශතයක් ලෙස
        - සරල ලාභාංශ ගණනය

        MEDIUM සඳහා:
        - ලාභාංශ + ප්‍රාග්ධන ලාභ ගණනය
        - ප්‍රතිශත ආදායම සොයන්න
        - කොටස් විකිණීමේ සන්දර්භය භාවිතා කරන්න

        HARD සඳහා:
        - සමාගම් දෙකක් හෝ වැඩි ගණනක් ඇතුළත් කරන්න
        - x භාවිතයෙන් සමීකරණයක් ගොඩනගන්න
        - අවසානයේ අපේක්ෂිත ප්‍රතිශත ලාභය ඉටු වූදැයි තර්ක කරන්න

        අවසාන පිළිතුර:
        - කොටස්: 5000 කොටස්
        - මුදල: රු. 54,000
        - ප්‍රතිශතය: 12.5%
        - තර්කය: “20% < 17.7% නිසා අපේක්ෂාව ඉටු වී නැත” වැනි ආකාරයෙන්
        """
        },
            
            'ලඝුගණක': {
        'difficulty': {

            'easy': {
                'steps': '3-6',
                'description': 'basic indices, fractional indices and simple exponential equations',
                'numbers': 'පූර්ණ සංඛ්‍යා, භාග, සරල දශම',
                'context': 'බල, මූල, භාගීය දර්ශක',
                'sub_topics': [
                    'බලයක භාගීය දර්ශක',
                    'බල හා මූල සරල කිරීම',
                    'සරල දර්ශක සමීකරණ'
                ],
                'examples': [
                    '³√27 = 27^(1/3) ලෙස ලිවීම',
                    '(√25)² සරල කිරීම',
                    '(27/64)^(2/3) අගය සොයීම',
                    '4ˣ = 64 වැනි දර්ශක සමීකරණ'
                ]
            },

            'medium': {
                'steps': '6-10',
                'description': 'logarithm laws, exponential equations and characteristic–mantissa handling',
                'numbers': 'දශම, ඍණ ලඝුගණක',
                'context': 'log නීති, lg, logₐ, විශාලය හා අතුළත',
                'sub_topics': [
                    'ලඝුගණක නීති (product, quotient, power)',
                    'logarithmic equations විසඳීම',
                    'විශාලය (Characteristic) හා අතුළත (Mantissa)',
                    'ලඝුගණක එකතු කිරීම හා අඩු කිරීම'
                ],
                'examples': [
                    'lg1000, log₄√64 ගණනය',
                    '2 log₂3 + 3 log₂2 − log₂72',
                    '2̄.5143 + 1̄.2375 වැනි එකතු කිරීම්',
                    'lg x සොයා x = 25 වැනි ප්‍රශ්න'
                ]
            },

            'hard': {
                'steps': '10-20',
                'description': 'log tables, powers, roots, complex expressions and real applications',
                'numbers': 'විශාල හා ඉතා කුඩා දශම',
                'context': 'ලඝුගණක වගු, antilog, scientific notation',
                'sub_topics': [
                    'ලඝුගණක වගු භාවිතයෙන් ගුණ හා බෙදීම',
                    'බල හා මූල log භාවිතයෙන් සෙවීම',
                    'සංකීර්ණ ප්‍රකාශන සුළු කිරීම',
                    'ලඝුගණක වල භාවිත (භෞතික / ජ්‍යාමිතීය)'
                ],
                'examples': [
                    '43.85 × 0.7532 (log table)',
                    '0.0875 ÷ 18.75 (negative characteristic)',
                    '√8.75, ³√0.9371 (antilog)',
                    '(7.543 × 0.987²) / √0.875',
                    'V = 4/3 πr³ යොදා ගෝල පරිමාව'
                ],
                'formulas': [
                    'log(ab) = log a + log b',
                    'log(a/b) = log a − log b',
                    'log aⁿ = n log a',
                    'antilog(log x) = x',
                    'a = 10^(characteristic + mantissa)'
                ]
            }
        },

        'prompt_template': """ප්‍රශ්නයේ සන්දර්භය:
    - A/L මට්ටමේ ලඝුගණක හා දර්ශක පාඩමට අදාළ විය යුතුය
    - lg, logₐ, antilog සංකේත නිවැරදිව භාවිතා කරන්න
    - log tables භාවිතා කරන විට characteristic හා mantissa වෙන් කර පෙන්වන්න

    EASY:
    - භාගීය දර්ශක හා මූල
    - සරල exponential equations
    - 3–6 පියවර

    MEDIUM:
    - log laws භාවිතයෙන් සරල කිරීම
    - lg x සොයාගැනීම
    - negative characteristic සහිත එකතු / අඩු කිරීම

    HARD:
    - log tables භාවිතයෙන් ගුණ, බෙදීම
    - බල හා මූල log භාවිතයෙන් සෙවීම
    - සංකීර්ණ ප්‍රකාශන
    - භෞතික / ජ්‍යාමිතීය භාවිත

    අවසාන පිළිතුර:
    - log අගය: 1̄.5179
    - antilog අගය: 33.03
    - ආසන්න අගය: දශම්ශ 1 හෝ 2 දක්වා
    - අවසාන තර්කය පැහැදිලිව සඳහන් කරන්න
    """
    },
            
            'ශ්‍රීඝ්‍රතාවය': {
    'difficulty': {
        'easy': {
            'steps': '2-3',
            'description': 'basic understanding of speed using simple values',
            'numbers': '1-100',
            'context': 'ශ්‍රීඝ්‍රතාවයේ මූලික සංකල්ප',
            'sub_topics': [
                'ශ්‍රීඝ්‍රතාවය යනු කුමක්ද',
                'දුර, කාලය, ශ්‍රීඝ්‍රතාවය අතර සම්බන්ධය',
                'සරල ගණනය'
            ],
            'examples': [
                'මෝටර් රථයක් පැය 2ක් තුළ km 60ක් ගමන් කරයි. ශ්‍රීඝ්‍රතාවය සොයන්න',
                'පදිකයෙක් පැය 1ක් තුළ km 5ක් ගමන් කරයි'
            ]
        },

        'medium': {
            'steps': '4-6',
            'description': 'unit conversions and multi-step speed problems',
            'numbers': '1-500',
            'context': 'ශ්‍රීඝ්‍රතාවය ගණනය සහ ඒකක පරිවර්තනය',
            'sub_topics': [
                'km/h ↔ m/s පරිවර්තනය',
                'දුර හෝ කාලය සොයාගැනීම',
                'බහු පියවර ගැටළු'
            ],
            'examples': [
                '72 km/h m/s බවට පරිවර්තනය කරන්න',
                'm/s 10 km/h බවට පරිවර්තනය කරන්න',
                'ශ්‍රීඝ්‍රතාවය 60 km/h නම් පැය 3ක දුර සොයන්න'
            ]
        },

        'hard': {
            'steps': '6-10',
            'description': 'complex word problems involving speed, time and distance',
            'numbers': 'ආසන්න වශයෙන් 1-1000',
            'context': 'ශ්‍රීඝ්‍රතාවය යෙදවුම් ගැටළු',
            'sub_topics': [
                'දෛනික ජීවිත ගැටළු',
                'විවිධ ඒකක සමඟ ගණනය',
                'O/L exam-style problems'
            ],
            'examples': [
                'දුම්රියක් 90 km/h ශ්‍රීඝ්‍රතාවයෙන් පැය 2½ ගමන් කරයි. ගමන් කළ දුර සොයන්න',
                'කාර් එකක් m/s 20 ශ්‍රීඝ්‍රතාවයෙන් ගමන් කරයි. km/h බවට පරිවර්තනය කරන්න',
                'දෙදෙනාගේ ශ්‍රීඝ්‍රතාවය සසඳා බැලීම'
            ],
            'formulas': [
                'ශ්‍රීඝ්‍රතාවය = දුර / කාලය',
                'දුර = ශ්‍රීඝ්‍රතාවය × කාලය',
                'කාලය = දුර / ශ්‍රීඝ්‍රතාවය',
                'km/h → m/s = × 5/18',
                'm/s → km/h = × 18/5'
            ]
        },
        
    },

    'prompt_template': """ප්‍රශ්නයේ සන්දර්භය:
- දුර, කාලය සහ ශ්‍රීඝ්‍රතාවය අතර සම්බන්ධය භාවිතා කරන්න
- නිවැරදි සූත්‍රය තෝරාගන්න
- ඒකක පරිවර්තනය අවශ්‍ය නම් සිදු කරන්න
- පියවරෙන් පියවර විසඳුම සිංහලෙන් පැහැදිලි කරන්න

EASY සඳහා:
- සරල සංඛ්‍යා භාවිතා කරන්න
- km/h පමණක් භාවිතා කරන්න
- එක් සූත්‍රයක් පමණක් යොදන්න

MEDIUM සඳහා:
- km/h ↔ m/s පරිවර්තනය ඇතුළත් කරන්න
- දුර හෝ කාලය සොයාගන්න
- පියවර 2–3ක් භාවිතා කරන්න

HARD සඳහා:
- දෛනික ජීවිත ගැටළු භාවිතා කරන්න
- විවිධ ඒකක මිශ්‍ර කරන්න
- O/L විභාග ශෛලියේ ප්‍රශ්න

අවසාන පිළිතුර:
- අගය + නිවැරදි ඒකක (km/h, m/s)
- අවශ්‍ය නම් වටකුරු (rounding) පැහැදිලි කරන්න"""
},
            
            'සමාන්තර ශ්‍රේණි': {
    'difficulty': {
        'easy': {
            'steps': '3-5',
            'description': 'identifying arithmetic progressions and finding nth term',
            'numbers': '1-100',
            'context': 'සමාන්තර ශ්‍රේඪියේ මූලික සංකල්ප',
            'sub_topics': [
                'සමාන්තර ශ්‍රේඪිය හඳුනාගැනීම',
                'මුල් පදය (a) සහ පොදු අන්තරය (d)',
                'n වන පදය (Tₙ)'
            ],
            'examples': [
                '2, 5, 8, 11,… යනු සමාන්තර ශ්‍රේඪියක් බව පෙන්වන්න',
                'a = 3, d = 4 නම් T₁₀ සොයන්න'
            ]
        },

        'medium': {
            'steps': '5-8',
            'description': 'finding number of terms and sum of arithmetic progressions',
            'numbers': '1-500',
            'context': 'n වන පදය හා ඓක්‍යය ගණනය',
            'sub_topics': [
                'මුල් පද n හි ඓක්‍යය (sₙ)',
                'n සොයාගැනීම',
                'a, d, l අතර සම්බන්ධය'
            ],
            'examples': [
                'a = 2, d = 3 නම් මුල් පද 20 හි ඓක්‍යය සොයන්න',
                'Tₙ = 62 නම් n සොයන්න'
            ]
        },

        'hard': {
            'steps': '8-15',
            'description': 'complex word problems and simultaneous equations',
            'numbers': 'විචල්‍ය සහිත අගයන්',
            'context': 'සංකීර්ණ සමාන්තර ශ්‍රේඪි ගැටලු',
            'sub_topics': [
                'වචන ගැටලු',
                'සමගාමී සමීකරණ සමඟ ශ්‍රේඪි',
                'O/L exam-style problems'
            ],
            'examples': [
                'මුල් පදය සහ පොදු අන්තරය සම්බන්ධ සමීකරණ දෙකක් විසඳීම',
                'ඓක්‍යය දී ඇති විට පද ගණන සොයන ගැටලු'
            ],
            'formulas': [
                'Tₙ = a + (n − 1)d',
                'sₙ = (n/2){2a + (n − 1)d}',
                'sₙ = (n/2)(a + l)'
            ]
        }
    }
}
        }
    
    def add_topic_config(self, topic: str, config: Dict):
        """Add or update topic configuration"""
        self.topic_configs[topic] = config
        print(f"Added/updated configuration for topic: {topic}")
    
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
            print("ChromaDB initialized with multilingual embeddings")
            
        except ImportError as e:
            print(f"ChromaDB not available: {e}")
            print("Install with: pip install chromadb sentence-transformers")
        except Exception as e:
            print(f"ChromaDB setup error: {e}")
    
    def _ensure_model(self):
        """Lazy load the Gemini model"""
        if self.model is None:
            print(f"Loading model: {self.model_name}")
            self.model = genai.GenerativeModel(self.model_name)
    
    def _rate_limit_wait(self):
        """Implement rate limiting for free tier"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            wait_time = self.min_request_interval - elapsed
            print(f"Rate limit: waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    # ==================== Data Loading ====================
    
    def load_all_data(
        self,
        examples_path: str = "data/extracted_text/extracted_examples.json",
        exercises_path: str = "data/extracted_text/exteacted_exercises.json",
        paragraphs_path: str = "data/extracted_text/paragraphs_and_tables.json",
        guidelines_path: str = "data/extracted_text/guidelines.json"
    ) -> bool:
        """Load all data files into ChromaDB"""
        if not self.chroma_client:
            print("ChromaDB not available, skipping data loading")
            return False
        
        print("\n" + "=" * 60)
        print("LOADING RAG DATA")
        print("=" * 60)
        
        # Setup collections
        self._setup_collections()
        
        # Load each data source
        paths = {
            'examples': examples_path,
            'exercises': exercises_path,
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
                    print(f"Error loading {name}: {e}")
            else:
                print(f"File not found: {path}")
        
        self.data_loaded = loaded_count > 0
        print(f"\nData loading complete: {loaded_count} sources loaded")
        return self.data_loaded
    
    def _setup_collections(self):
        """Create ChromaDB collections"""
        collection_names = {
            'examples': 'sinhala_examples',
            'exercises': 'sinhala_exercises',
            'paragraphs': 'sinhala_paragraphs',
            'guidelines': 'sinhala_guidelines'
        }
        
        for key, name in collection_names.items():
            try:
                self.collections[key] = self.chroma_client.get_collection(
                    name=name,
                    embedding_function=self.embedding_fn
                )
                print(f"♻️ Using existing collection: {name}")
            except Exception:
                try:
                    self.collections[key] = self.chroma_client.create_collection(
                        name=name,
                        embedding_function=self.embedding_fn
                    )
                    print(f"✨ Created collection: {name}")
                except Exception as e:
                    print(f"Failed to create {name}: {e}")
    
    def _load_data_file(self, name: str, path: str):
        """Load a specific data file into ChromaDB - handles various structures"""
        print(f"📂 Loading {name} from {path}...")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ❌ Error reading file {path}: {e}")
            return
        
        texts, metadata_list, ids = [], [], []
        
        if name == 'examples':
            # Handle examples - could be list or dict with 'examples' key
            examples = data if isinstance(data, list) else data.get('examples', [])
            
            for i, example in enumerate(examples):
                if not isinstance(example, dict):
                    print(f"  ⚠️ Skipping non-dict example at index {i}")
                    continue
                    
                # Build full text from structure
                q_text = example.get('question', example.get('Question', ''))
                full_text = f"උදාහරණය:\n{q_text}\n\nවිසඳුම:\n"
                
                # Handle steps
                steps = example.get('Steps', example.get('steps', []))
                for step in steps:
                    if isinstance(step, dict):
                        step_text = step.get('step_answer', step.get('Step', step.get('step', '')))
                        full_text += f"{step_text}\n"
                    elif isinstance(step, str):
                        full_text += f"{step}\n"
                
                final_ans = example.get('Final_answer', example.get('final_answer', ''))
                full_text += f"\nඅවසාන පිළිතුර: {final_ans}"
                
                texts.append(full_text)
                meta = {
                    'type': 'example',
                    'index': i,
                    'topic': str(example.get('topic', '')),
                    'sub_topic': str(example.get('sub_topic', ''))
                }
                metadata_list.append(meta)
                ids.append(f"ex_{i}")
            
            self.data['examples'] = examples
            
        elif name == 'exercises':
            # Handle exercises - could be list or dict with 'exercises' key
            exercises_raw = data if isinstance(data, list) else data.get('exercises', [])
            
            # If still not a list, try other common keys
            if not isinstance(exercises_raw, list):
                for key in ['exercise', 'questions', 'data']:
                    if key in data and isinstance(data[key], list):
                        exercises_raw = data[key]
                        break
            
            if not isinstance(exercises_raw, list):
                print(f"  ⚠️ Could not find exercises list. Keys: {data.keys() if isinstance(data, dict) else 'N/A'}")
                exercises_raw = []
            
            exercises = []
            for i, exercise in enumerate(exercises_raw):
                # Skip if not a dictionary
                if not isinstance(exercise, dict):
                    print(f"  ⚠️ Skipping non-dict exercise at index {i}: {type(exercise)}")
                    continue
                
                exercises.append(exercise)
                
                # ===== Handle BOTH structures =====
                
                # Structure 1: Direct 'question' key
                main_q = exercise.get('question', '')
                
                # Structure 2: 'text' key or nested in 'metadata'
                if not main_q:
                    main_q = exercise.get('text', '')
                if not main_q:
                    metadata_obj = exercise.get('metadata', {})
                    if isinstance(metadata_obj, dict):
                        main_q = metadata_obj.get('main_question', '')
                
                # Build full text
                full_text = f"අභ්‍යාස ප්‍රශ්නය:\n{main_q}"
                
                # ===== Handle sub_questions from BOTH structures =====
                sub_qs = []
                
                # Structure 1: Direct 'sub_questions' key
                direct_sub_qs = exercise.get('sub_questions', [])
                if isinstance(direct_sub_qs, list):
                    sub_qs = direct_sub_qs
                
                # Structure 2: Nested in 'metadata'
                if not sub_qs:
                    metadata_obj = exercise.get('metadata', {})
                    if isinstance(metadata_obj, dict):
                        nested_sub_qs = metadata_obj.get('sub_questions', [])
                        if isinstance(nested_sub_qs, list):
                            sub_qs = nested_sub_qs
                
                # Add sub-questions to text
                if sub_qs:
                    full_text += "\n\nඅනු ප්‍රශ්න:\n"
                    for j, sub in enumerate(sub_qs, 1):
                        if isinstance(sub, dict):
                            # Handle both 'sub_question' and 'question' keys
                            q_text = sub.get('sub_question', sub.get('question', sub.get('text', '')))
                            if q_text:
                                full_text += f"{j}. {q_text}\n"
                        elif isinstance(sub, str):
                            full_text += f"{j}. {sub}\n"
                
                # Get topic - handle both structures
                topic = exercise.get('topic', '')
                sub_topic = exercise.get('sub_topic', '')
                
                texts.append(full_text)
                meta = {
                    'type': 'exercise',
                    'index': i,
                    'topic': str(topic) if topic else '',
                    'sub_topic': str(sub_topic) if sub_topic else ''
                }
                metadata_list.append(meta)
                ids.append(f"exr_{i}")
            
            self.data['exercises'] = exercises
            print(f"  📊 Processed {len(exercises)} exercises")
            
        elif name == 'paragraphs':
            # Handle paragraphs
            paragraphs_raw = data if isinstance(data, list) else data.get('paragraphs', [])
            
            paragraphs = []
            for i, para in enumerate(paragraphs_raw):
                if isinstance(para, dict):
                    text_content = para.get('text', para.get('content', ''))
                    paragraphs.append(para)
                    topic = para.get('topic', '')
                    page = para.get('page')
                elif isinstance(para, str):
                    text_content = para
                    paragraphs.append({'text': para})
                    topic = ''
                    page = None
                else:
                    continue
                
                if text_content:
                    texts.append(text_content)
                    meta = {
                        'type': 'paragraph',
                        'page': page,
                        'topic': str(topic) if topic else ''
                    }
                    metadata_list.append(meta)
                    ids.append(f'para_{i}')
            
            self.data['paragraphs'] = paragraphs
            
        elif name == 'guidelines':
            # Handle guidelines - can be nested or flat
            guidelines_raw = data if isinstance(data, list) else data.get('guideline', data.get('guidelines', []))
            
            guideline_idx = 0
            
            for item in guidelines_raw:
                if isinstance(item, dict):
                    # Nested structure with topic and content
                    topic = item.get('topic', '')
                    content_list = item.get('content', [])
                    
                    if isinstance(content_list, list):
                        for content in content_list:
                            if isinstance(content, str) and content.strip():
                                texts.append(content)
                                metadata_list.append({
                                    'type': 'guideline',
                                    'index': guideline_idx,
                                    'topic': str(topic) if topic else ''
                                })
                                ids.append(f"guide_{guideline_idx}")
                                guideline_idx += 1
                    elif isinstance(content_list, str) and content_list.strip():
                        texts.append(content_list)
                        metadata_list.append({
                            'type': 'guideline',
                            'index': guideline_idx,
                            'topic': str(topic) if topic else ''
                        })
                        ids.append(f"guide_{guideline_idx}")
                        guideline_idx += 1
                        
                elif isinstance(item, str) and item.strip():
                    # Flat structure - just strings
                    texts.append(item)
                    metadata_list.append({
                        'type': 'guideline',
                        'index': guideline_idx,
                        'topic': ''
                    })
                    ids.append(f"guide_{guideline_idx}")
                    guideline_idx += 1
            
            self.data['guidelines'] = guidelines_raw
        
        # Add to collection
        if texts and name in self.collections:
            try:
                self.collections[name].add(
                    documents=texts,
                    metadatas=metadata_list,
                    ids=ids
                )
                print(f"  ✅ Loaded {len(texts)} {name}")
            except Exception as e:
                print(f"  �� Error adding to collection: {e}")
        elif not texts:
            print(f"  ⚠️ No valid {name} found to load")
    
    # ==================== Context Retrieval ====================
    
    def retrieve_context(
        self,
        query: str,
        topic: str = None,
        n_results: int = 3
    ) -> Dict[str, List[Dict]]:
        """
        Retrieve relevant context from all collections
        Filter by topic if specified
        """
        results = {}
        
        if not self.collections:
            return results
        
        for name, collection in self.collections.items():
            try:
                # Build where filter for topic
                where_filter = None
                if topic:
                    where_filter = {"topic": topic}
                
                search = collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where_filter
                )
                
                items = []
                if search.get('documents') and search['documents'][0]:
                    for i in range(len(search['documents'][0])):
                        items.append({
                            'text': search['documents'][0][i],
                            'distance': search['distances'][0][i] if search.get('distances') else 0,
                            'metadata': search['metadatas'][0][i] if search.get('metadatas') else {}
                        })
                results[name] = items
                
            except Exception as e:
                print(f"Error querying {name}: {e}")
                results[name] = []
        
        return results
    
    # ==================== Unified Prompt Building ====================
    
    def _build_prompt_with_context(
        self,
        topic: str,
        difficulty: str,
        num_questions: int,
        context: Dict,
        existing_count: int = 0
    ) -> str:
        """
        Unified prompt builder - works for all topics
        Uses topic-specific configurations
        """
        # Get topic config or use default
        topic_config = self.topic_configs.get(topic)
        
        if not topic_config:
            print(f"No configuration for topic '{topic}', using default")
            topic_config = self.topic_configs.get('පොළිය', {})
        
        # Get difficulty config
        diff_configs = topic_config.get('difficulty', {})
        config = diff_configs.get(difficulty, diff_configs.get('medium', {}))
        
        start_num = existing_count + 1
        
        # Build context section from RAG results
        context_section = ""
        
        if context.get('examples'):
            context_section += "\nREFERENCE EXAMPLES (use similar format):\n"
            for i, ex in enumerate(context['examples'][:2], 1):
                context_section += f"\nExample {i}:\n{ex['text'][:500]}...\n"
        
        if context.get('guidelines'):
            context_section += "\n📋 GUIDELINES:\n"
            for guide in context['guidelines'][:2]:
                context_section += f"- {guide['text'][:200]}\n"
        
        # Get topic-specific prompt template
        topic_template = topic_config.get('prompt_template', '')
        
        # Build complete prompt
        prompt = f"""You are an expert O/L mathematics teacher creating questions in Sinhala.

TOPIC: {topic}
DIFFICULTY: {difficulty} ({config.get('description', 'standard problems')})
STEPS: {config.get('steps', '3-4')}
NUMBER RANGE: {config.get('numbers', 'විචල්‍ය')}
CONTEXT: {config.get('context', '')}

{context_section}

{topic_template}

IMPORTANT: Generate ALL {num_questions} complete questions. Do NOT stop early.

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

ANSWER: [Final answer]

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
✓ Separate each question with ---
✓ Include SOLUTION and ANSWER for each
✓ Follow the topic-specific guidelines above

Generate {num_questions} questions about {topic}:
"""
        return prompt
    
    # ==================== Response Parsing (unchanged) ====================
    
    def _parse_response(self, text: str) -> List[Dict]:
        """Parse generated questions from Gemini response"""
        print(f"\nResponse: {len(text)} chars")
        
        questions = []
        
        # Split by separator
        if '---' in text:
            parts = text.split('---')
            parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 50]
        else:
            parts = re.split(r'(?=QUESTION\s*\d+\s*:)', text, flags=re.IGNORECASE)
            parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 50]
        
        print(f"Found {len(parts)} sections")
        
        for part in parts:
            question_data = self._extract_question(part)
            if question_data:
                questions.append(question_data)
                print(f"Question {len(questions)} parsed")
        
        return questions
    
    def _extract_question(self, section: str) -> Optional[Dict]:
        """Extract question, solution, and answer from a section"""
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
        
        # Fallback: get last calculation result as answer
        if a_text == "N/A" and s_text:
            lines = s_text.strip().split('\n')
            for line in reversed(lines):
                if '=' in line and any(c.isdigit() for c in line):
                    a_text = line.split('=')[-1].strip()
                    break
        
        # Clean question text
        if q_text:
            q_text = re.sub(r'\s+', ' ', q_text).strip()
            q_text = re.sub(r'\s*(SOLUTION|විසඳුම)\s*:?\s*$', '', q_text, flags=re.IGNORECASE).strip()
        
        # Validate
        if q_text and s_text and len(q_text) > 20 and len(s_text) > 20:
            return {
                'question': q_text,
                'solution': s_text,
                'answer': a_text
            }
        
        return None
    
    # ==================== Main Generation ====================
    
    def generate_questions(
        self,
        topic: str,
        difficulty: str,
        num_questions: int
    ) -> Tuple[List[Dict], bool]:
        """Generate questions using RAG context with topic-aware retrieval"""
        print(f"\n{'='*60}")
        print(f"GENERATING {num_questions} QUESTIONS WITH RAG")
        print(f"{'='*60}")
        print(f"Topic: {topic}")
        print(f"Difficulty: {difficulty}")
        print(f"Model: {self.model_name}")
        print(f"RAG Data Loaded: {self.data_loaded}")
        
        # Validate topic
        if topic not in self.topic_configs:
            print(f"Warning: Topic '{topic}' not in configurations")
        
        self._ensure_model()
        
        # Retrieve context using RAG - with topic filter
        context = {}
        rag_used = False
        
        if self.data_loaded and self.collections:
            print("\nRetrieving RAG context...")
            context = self.retrieve_context(
                f"{topic} උදාහරණ ප්‍රශ්න",
                topic=topic,
                n_results=3
            )
            rag_used = any(len(items) > 0 for items in context.values())
            
            if rag_used:
                total_context = sum(len(items) for items in context.values())
                print(f"Retrieved {total_context} context items for topic '{topic}'")
            else:
                print(f"No relevant context found for topic '{topic}'")
        
        all_questions = []
        max_attempts = 5
        attempt = 0
        
        while len(all_questions) < num_questions and attempt < max_attempts:
            attempt += 1
            remaining = num_questions - len(all_questions)
            
            print(f"\nAttempt {attempt}/{max_attempts} - Need {remaining} questions...")
            
            try:
                self._rate_limit_wait()
                
                request_count = min(remaining + 2, 7)
                
                prompt = self._build_prompt_with_context(
                    topic, difficulty, request_count, context, len(all_questions)
                )
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                
                if not response.text:
                    print("Empty response")
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
                    
                    print(f" Progress: {len(all_questions)}/{num_questions}")
                    
                    if len(all_questions) >= num_questions:
                        break
                else:
                    print("No questions parsed")
                
                if len(all_questions) < num_questions:
                    time.sleep(2)
                    
            except Exception as e:
                error_str = str(e).lower()
                print(f"Error: {str(e)[:100]}")
                
                if "quota" in error_str or "rate" in error_str:
                    wait_time = attempt * 10
                    print(f" Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    time.sleep(5)
        
        if all_questions:
            print(f"\n{'='*60}")
            print(f"Generated {len(all_questions)}/{num_questions} questions")
            print(f"RAG Context Used: {rag_used}")
            print(f"{'='*60}")
            return all_questions[:num_questions], rag_used
        
        raise Exception("Failed to generate questions. Please try again.")
    
    # ==================== Utility Methods ====================
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics about loaded collections"""
        stats = {}
        for name, collection in self.collections.items():
            try:
                stats[name] = collection.count()
            except:
                stats[name] = 0
        return stats
    
    def get_available_topics(self) -> List[str]:
        """Get list of configured topics"""
        return list(self.topic_configs.keys())
    
    def export_questions(self, questions: List[Dict], path: str = "generated_questions.json"):
        """Export generated questions to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)
        print(f" Saved {len(questions)} questions to: {path}")


# ==================== Standalone Usage ====================

if __name__ == "__main__":
    """Test the RAG system with multiple topics"""
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("GEMINI_API_KEY not found in .env file")
        exit(1)
    
    print("\n" + "=" * 70)
    print("🎓 SINHALA MATH RAG SYSTEM - MULTI-TOPIC TEST")
    print("=" * 70)
    
    # Initialize RAG system
    rag = SinhalaRAGSystem(api_key)
    
    # Load data
    print("\nLoading all data sources...")
    rag.load_all_data()
    
    # Show collection stats
    print("\n Collection Statistics:")
    stats = rag.get_collection_stats()
    for name, count in stats.items():
        print(f"{name}: {count} items")
    
    # Show available topics
    print(f"\nAvailable Topics: {', '.join(rag.get_available_topics())}")
    
    # Test 1: Generate පොළිය questions
    print("\n" + "=" * 70)
    print("TEST 1: Generating පොළිය questions")
    print("=" * 70)
    
    questions_poliya, rag_used = rag.generate_questions(
        topic="පොළිය",
        difficulty="medium",
        num_questions=2
    )
    
    print(f"\n Results for පොළිය:")
    print(f"Generated: {len(questions_poliya)} questions")
    print(f"RAG Used: {rag_used}")
    
    for i, q in enumerate(questions_poliya, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {q['question'][:150]}...")
        print(f"A: {q['answer']}")
    
    # Test 2: Generate සමීකරණ questions
    print("\n" + "=" * 70)
    print("TEST 2: Generating සමීකරණ questions")
    print("=" * 70)
    
    questions_equations, rag_used = rag.generate_questions(
        topic="සමීකරණ",
        difficulty="easy",
        num_questions=2
    )
    
    print(f"\n Results for සමීකරණ:")
    print(f"Generated: {len(questions_equations)} questions")
    print(f"RAG Used: {rag_used}")
    
    for i, q in enumerate(questions_equations, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {q['question'][:150]}...")
        print(f"A: {q['answer']}")
    
    # Export all questions
    all_questions = {
        'පොළිය': questions_poliya,
        'සමීකරණ': questions_equations
    }
    
    output_path = "multi_topic_questions.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    
    print(f"\n All questions saved to: {output_path}")
    print("\nMulti-topic test completed successfully!")