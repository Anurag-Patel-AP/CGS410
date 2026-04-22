import json
import os

class DataLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataLoader, cls).__new__(cls)
            cls._instance.train_data = {}  # used for specialist discovery (seen)
            cls._instance.test_data  = {}  # held-out for user inference (unseen)
        return cls._instance
        
    def load_all(self):
        langs = ["en", "hi", "ja", "de"]
        data_dir = os.path.join(os.path.dirname(__file__), "data")

        for lang in langs:
            # Training set (70% of sentences) — used for specialist discovery
            train_path = os.path.join(data_dir, f"{lang}.json")
            if os.path.exists(train_path):
                with open(train_path, "r", encoding="utf-8") as f:
                    self.train_data[lang] = json.load(f)
                print(f"Loaded {len(self.train_data[lang])} train sentences for {lang}.")
            else:
                print(f"Warning: Train data file for {lang} not found at {train_path}")
                self.train_data[lang] = []

            # Test set (30% held-out split) — never seen during specialist discovery
            test_path = os.path.join(data_dir, f"{lang}_test.json")
            if os.path.exists(test_path):
                with open(test_path, "r", encoding="utf-8") as f:
                    self.test_data[lang] = json.load(f)
                print(f"Loaded {len(self.test_data[lang])} test  sentences for {lang}. [HELD-OUT]")
            else:
                print(f"Warning: Test data file for {lang} not found. Falling back to train set for UI.")
                self.test_data[lang] = self.train_data[lang]

    def get_sentences(self, lang):
        """Returns train sentences (used during specialist discovery)."""
        return self.train_data.get(lang, [])

    def get_test_sentences(self, lang):
        """Returns held-out test sentences (shown in UI dropdown)."""
        return self.test_data.get(lang, [])

    def get_test_sentence(self, lang, sentence_id):
        """Fetch one held-out sentence by positional index."""
        sentences = self.get_test_sentences(lang)
        if 0 <= sentence_id < len(sentences):
            return sentences[sentence_id]
        # fallback: search by id field
        for s in sentences:
            if s.get("id") == sentence_id:
                return s
        return None

data_manager = DataLoader()
