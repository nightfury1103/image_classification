import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

criterion_class = nn.CrossEntropyLoss()
criterion_desc = nn.CrossEntropyLoss()  # For simplicity, adjust for sequence generation

