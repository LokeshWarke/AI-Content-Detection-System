"""
Enhanced AI Content Detector with multiple model options:
1. Transformer-based (BERT/RoBERTa) - Best accuracy, moderate speed
2. Lightweight LLM (Mistral/Llama via Ollama) - Excellent accuracy, slower
3. Original TF-IDF (fallback) - Fast, lower accuracy
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Try importing transformer models
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available. Install with: pip install transformers torch")

# Try importing Ollama for local LLM support
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  Ollama not available. Install with: pip install ollama")

# Original imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import nltk
from nltk.corpus import brown
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
import math
import re
from collections import Counter

# Ensure NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown')


class TransformerDetector:
    """Transformer-based detection using BERT or RoBERTa"""
    
    def __init__(self, model_name: str = "roberta-base"):
        """
        Initialize transformer detector.
        Options:
        - 'roberta-base': Fast, good accuracy (~110M parameters)
        - 'distilroberta-base': Faster, slightly lower accuracy (~82M parameters)
        - 'bert-base-uncased': Alternative option
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")
        
        self.model_name = model_name
        self.device = "cpu"  # Use GPU if available: "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        try:
            if "roberta" in model_name.lower():
                self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
                self.model = RobertaForSequenceClassification.from_pretrained(
                    model_name, 
                    num_labels=2
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=2
                )
            
            # Create pipeline for easier inference
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1 if self.device == "cpu" else 0,
                return_all_scores=True
            )
            
            print(f"✅ Loaded transformer model: {model_name}")
        except Exception as e:
            print(f"❌ Error loading transformer model: {e}")
            raise
    
    def predict(self, text: str) -> float:
        """Get AI probability score (0-1)"""
        try:
            # Truncate to model's max length (typically 512 tokens)
            results = self.classifier(text[:1000])  # Rough truncation
            if isinstance(results, list) and len(results) > 0:
                # Find AI label (usually label 1 or "LABEL_1")
                for result in results[0]:
                    if result['label'] in ['LABEL_1', '1', 'AI']:
                        return result['score']
                # If no AI label found, return complement of human label
                return 1 - results[0][0]['score']
            return 0.5
        except Exception as e:
            print(f"Error in transformer prediction: {e}")
            return 0.5


class OllamaLLMDetector:
    """Detection using local LLM via Ollama (Mistral, Llama, etc.)"""
    
    def __init__(self, model_name: str = "mistral:7b"):
        """
        Initialize Ollama detector.
        Requires Ollama to be installed and running locally.
        
        Model options:
        - 'mistral:7b' - Fast, good accuracy (~4GB RAM)
        - 'llama3:8b' - Alternative option
        - 'mistral:7b-instruct' - Instruction-tuned version
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama library not available. Install with: pip install ollama")
        
        self.model_name = model_name
        self.client = ollama.Client()
        
        # Test connection
        try:
            models = self.client.list()
            print(f"✅ Connected to Ollama. Available models: {[m['name'] for m in models.get('models', [])]}")
        except Exception as e:
            print(f"⚠️  Warning: Could not connect to Ollama: {e}")
            print("   Make sure Ollama is installed and running: https://ollama.ai")
    
    def predict(self, text: str) -> float:
        """Get AI probability using LLM"""
        try:
            prompt = f"""Analyze the following text and determine if it was written by a human or generated by AI. 
Respond with only a number between 0 and 1, where:
- 0.0-0.3 = Definitely human-written
- 0.3-0.5 = Likely human-written
- 0.5-0.7 = Likely AI-generated
- 0.7-1.0 = Definitely AI-generated

Text: {text[:2000]}

Probability (0-1):"""
            
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Low temperature for consistent results
                    "num_predict": 10  # Short response
                }
            )
            
            # Extract number from response
            result_text = response.get('response', '').strip()
            # Try to extract a float
            import re
            numbers = re.findall(r'0?\.\d+|1\.0|0', result_text)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
            
            return 0.5
        except Exception as e:
            print(f"Error in Ollama prediction: {e}")
            return 0.5


class EnhancedAIDetector:
    """Enhanced detector with multiple model options"""
    
    def __init__(self, method: str = "transformer"):
        """
        Initialize enhanced detector.
        
        Methods:
        - 'transformer': Use BERT/RoBERTa (best accuracy, moderate speed)
        - 'ollama': Use local LLM via Ollama (excellent accuracy, slower)
        - 'hybrid': Combine transformer + original methods
        - 'original': Use original TF-IDF method (fast, lower accuracy)
        """
        self.method = method
        self.transformer_detector = None
        self.ollama_detector = None
        self.original_detector = None
        
        # Initialize based on method
        if method == "transformer" and TRANSFORMERS_AVAILABLE:
            try:
                self.transformer_detector = TransformerDetector("distilroberta-base")
            except Exception as e:
                print(f"⚠️  Transformer model failed, falling back to original: {e}")
                method = "original"
        
        elif method == "ollama" and OLLAMA_AVAILABLE:
            try:
                self.ollama_detector = OllamaLLMDetector("mistral:7b")
            except Exception as e:
                print(f"⚠️  Ollama model failed, falling back to original: {e}")
                method = "original"
        
        elif method == "hybrid":
            # Use both transformer and original
            if TRANSFORMERS_AVAILABLE:
                try:
                    self.transformer_detector = TransformerDetector("distilroberta-base")
                except:
                    pass
            # Always have original as fallback
            self._init_original_detector()
        
        if method == "original" or not (self.transformer_detector or self.ollama_detector):
            self._init_original_detector()
    
    def _init_original_detector(self):
        """Initialize original TF-IDF detector"""
        from core import AIContentDetectorCore
        self.original_detector = AIContentDetectorCore()
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Get prediction from selected method(s)"""
        results = {}
        
        if self.method == "transformer" and self.transformer_detector:
            score = self.transformer_detector.predict(text)
            results = {
                "method": "transformer",
                "ai_probability": float(score),
                "human_probability": float(1 - score),
                "confidence": abs(score - 0.5) * 2  # 0-1 scale
            }
        
        elif self.method == "ollama" and self.ollama_detector:
            score = self.ollama_detector.predict(text)
            results = {
                "method": "ollama",
                "ai_probability": float(score),
                "human_probability": float(1 - score),
                "confidence": abs(score - 0.5) * 2
            }
        
        elif self.method == "hybrid":
            # Combine transformer and original
            scores = []
            
            if self.transformer_detector:
                trans_score = self.transformer_detector.predict(text)
                scores.append(("transformer", trans_score, 0.6))  # 60% weight
            
            if self.original_detector:
                orig_score = self.original_detector.classifier.predict_proba(
                    self.original_detector.vectorizer.transform([text])
                )[0, 1]
                scores.append(("original", orig_score, 0.4))  # 40% weight
            
            if scores:
                # Weighted average
                total_weight = sum(w for _, _, w in scores)
                combined_score = sum(s * w for _, s, w in scores) / total_weight
                
                results = {
                    "method": "hybrid",
                    "ai_probability": float(combined_score),
                    "human_probability": float(1 - combined_score),
                    "confidence": abs(combined_score - 0.5) * 2,
                    "components": {name: score for name, score, _ in scores}
                }
            else:
                # Fallback to original
                if self.original_detector:
                    score = self.original_detector.classifier.predict_proba(
                        self.original_detector.vectorizer.transform([text])
                    )[0, 1]
                    results = {
                        "method": "original",
                        "ai_probability": float(score),
                        "human_probability": float(1 - score),
                        "confidence": abs(score - 0.5) * 2
                    }
                else:
                    results = {
                        "method": "original",
                        "ai_probability": 0.5,
                        "human_probability": 0.5,
                        "confidence": 0.0
                    }
        
        else:
            # Original method
            if self.original_detector:
                score = self.original_detector.classifier.predict_proba(
                    self.original_detector.vectorizer.transform([text])
                )[0, 1]
                results = {
                    "method": "original",
                    "ai_probability": float(score),
                    "human_probability": float(1 - score),
                    "confidence": abs(score - 0.5) * 2
                }
        
        return results


# Example usage
if __name__ == "__main__":
    # Test different methods
    test_text = "The implementation of this system demonstrates a comprehensive approach to data analysis."
    
    print("Testing Enhanced AI Detector\n" + "="*50)
    
    # Test transformer (if available)
    if TRANSFORMERS_AVAILABLE:
        try:
            detector = EnhancedAIDetector("transformer")
            result = detector.predict(test_text)
            print(f"\n✅ Transformer Method:")
            print(f"   AI Probability: {result['ai_probability']:.2%}")
            print(f"   Human Probability: {result['human_probability']:.2%}")
            print(f"   Confidence: {result['confidence']:.2%}")
        except Exception as e:
            print(f"\n❌ Transformer failed: {e}")
    
    # Test original
    try:
        detector = EnhancedAIDetector("original")
        result = detector.predict(test_text)
        print(f"\n✅ Original Method:")
        print(f"   AI Probability: {result['ai_probability']:.2%}")
        print(f"   Human Probability: {result['human_probability']:.2%}")
        print(f"   Confidence: {result['confidence']:.2%}")
    except Exception as e:
        print(f"\n❌ Original failed: {e}")

