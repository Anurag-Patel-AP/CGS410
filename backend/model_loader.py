"""
Transformer Model Singleton Loader.
===================================

Provides a globally accessible singleton manager for Hugging Face pretrained models
and tokenizers. Prevents reloading massive multi-lingual transformer layers multiple 
times during web API requests, drastically improving latency.
"""
from transformers import AutoTokenizer, AutoModel

class ModelLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._tokenizer = None
            cls._instance._model = None
        return cls._instance
        
    def load_model(self, model_name="bert-base-multilingual-cased"):
        if self._tokenizer is None or self._model is None:
            print(f"Loading tokenizer and model: {model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name, attn_implementation="eager", output_attentions=True)
            print("Model loaded.")
            
    def get_tokenizer(self):
        return self._tokenizer
        
    def get_model(self):
        return self._model

# Create a singleton instance
model_manager = ModelLoader()
