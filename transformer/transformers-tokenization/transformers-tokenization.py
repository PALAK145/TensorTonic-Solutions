import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        sp_tokens = [self.pad_token,
                     self.unk_token,
                     self.bos_token,
                     self.eos_token]
        for i in range(4):
            self.word_to_id[sp_tokens[i]] = i
            self.id_to_word[i] = sp_tokens[i]
        words = set()
        for sentence in texts:
            words.update(sentence.lower().split())
        words = sorted(words)
        for i in range(len(words)):
            self.word_to_id[words[i]] = i + 4
            self.id_to_word[i + 4] = words[i]
        self.vocab_size = len(words) + 4
        pass
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        ids = []
        for word in text.lower().split():
            if word in self.word_to_id:
                ids.append(self.word_to_id[word])
            else:
                ids.append(1)
        return ids
        pass
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        words = []
        for id in ids:
            if id in self.id_to_word:
                words.append(self.id_to_word[id])
            else:
                words.append(self.unk_token)
        return " ".join(words)
        pass
