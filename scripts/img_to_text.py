import os, sys
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import tqdm
from sklearn.model_selection import train_test_split

curr_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_dir)
sys.path.append(os.path.join(curr_dir, '../'))

from data_utils import INTELDataset
from model_utils import MultiTaskModel

train_dataloader = INTELDataset(batch_size=32).get_train_dataloader()
valid_dataloader = INTELDataset(batch_size=32).get_valid_dataloader()
test_dataloader = INTELDataset(batch_size=32).get_test_dataloader()

vocab, vocab_size = INTELDataset().get_vocab()

# Printing vocab_size to verify
print(f"Vocabulary size, including special tokens: {vocab_size}")



best_valid_loss = float('inf')

# # Model, Loss, and Optimizer
model = MultiTaskModel(num_classes=6, vocab_size=vocab_size)
criterion_class = nn.CrossEntropyLoss()
criterion_desc = nn.CrossEntropyLoss()  # For simplicity, adjust for sequence generation
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10  # Specify the number of epochs

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, labels, captions in tqdm.tqdm(train_dataloader):
        optimizer.zero_grad()

            # Process all captions in the batch
        caption_indices_list = []
        for caption in captions:
            tokens = caption.split()
            token_indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
            caption_indices_list.append(token_indices)
        
        # Pad the sequences to have the same length
        caption_lengths = [len(indices) for indices in caption_indices_list]
        max_length = max(caption_lengths)
        padded_caption_indices = [indices + [vocab['<pad>']] * (max_length - len(indices)) for indices in caption_indices_list]

        # Convert the padded caption indices into a tensor
        caption_indices_tensor = torch.tensor(padded_caption_indices, dtype=torch.long)
        
        class_logits, description_logits = model(images, caption_indices_tensor)
        
        loss_class = criterion_class(class_logits, labels)
        loss_desc = criterion_desc(description_logits.view(-1, vocab_size), caption_indices_tensor.view(-1))
        
        loss = loss_class + 0.5 * loss_desc
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    # Validation step
    model.eval()
    valid_loss = 0
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for images, labels, captions in valid_dataloader:
            # Prepare validation captions
            tokens = captions[0].split()  # Assuming a batch size of 1 for simplicity
            token_indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
            caption_indices_tensor = torch.tensor(token_indices, dtype=torch.long).unsqueeze(0)
            
            class_logits, description_logits = model(images, caption_indices_tensor)
            
            loss_class = criterion_class(class_logits, labels)
            loss_desc = criterion_desc(description_logits.view(-1, vocab_size), caption_indices_tensor.view(-1))
            
            loss = loss_class + 0.5 * loss_desc
            valid_loss += loss.item()
            
            # Calculate accuracy
            _, predicted_labels = torch.max(class_logits, 1)
            correct_preds += (predicted_labels == labels).sum().item()
            total_preds += labels.size(0)

    # Average losses
    train_loss /= len(train_dataloader)
    valid_loss /= len(valid_dataloader)
    valid_acc = correct_preds / total_preds
    
    print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, Validation Acc: {valid_acc:.4f}')

    # Save model if it has the best validation loss
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best_model.pth')
        
# # # Test the model
# model.load_state_dict(torch.load('/home/huy/Desktop/HCMUS/image_classification/models/best_model.pth'))
# model.eval()
# test_loss = 0
# correct_preds = 0
# total_preds = 0
# with torch.no_grad():
#     for images, labels, captions in test_dataloader:
#         # Prepare test captions
#         tokens = captions[0].split()  # Assuming a batch size of 1 for simplicity
#         token_indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
#         caption_indices_tensor = torch.tensor(token_indices, dtype=torch.long).unsqueeze(0)
        
#         class_logits, description_logits = model(images, caption_indices_tensor)
        
#         loss_class = criterion_class(class_logits, labels)
#         loss_desc = criterion_desc(description_logits.view(-1, vocab_size), caption_indices_tensor.view(-1))
        
#         loss = loss_class + 0.5 * loss_desc
#         test_loss += loss.item()
        
#         # Calculate accuracy
#         _, predicted_labels = torch.max(class_logits, 1)
#         correct_preds += (predicted_labels == labels).sum().item()
#         total_preds += labels.size(0)

# test_loss /= len(test_dataloader)
# test_acc = correct_preds / total_preds
# print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
