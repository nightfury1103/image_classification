import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

criterion_class = nn.CrossEntropyLoss()
criterion_desc = nn.CrossEntropyLoss()  # For simplicity, adjust for sequence generation

def tokenize_and_pad_texts(tokens1: str, tokens2: str, padding_value: int = 0):
    # Determine the maximum length
    max_length = max(tokens1.shape[1], tokens2.shape[1])
    
    # Pad the tokenized texts to the same length
    if tokens1.shape[1] < max_length:
        tokens1 = torch.nn.functional.pad(tokens1, (0, max_length - tokens1.shape[1]), value=padding_value)
    if tokens2.shape[1] < max_length:
        tokens2 = torch.nn.functional.pad(tokens2, (0, max_length - tokens2.shape[1]), value=padding_value)
    
    return tokens1, tokens2