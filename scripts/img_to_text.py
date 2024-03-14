import os, sys
import torch
import torch.nn as nn

torch.manual_seed(42)

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
from torch.nn.utils.rnn import pad_sequence

curr_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_dir)
sys.path.append(os.path.join(curr_dir, '../'))

from data_utils import INTELDataset
from model_utils import MultiTaskModel

train_dataloader = INTELDataset(batch_size=64).get_train_dataloader()
valid_dataloader = INTELDataset(batch_size=64).get_valid_dataloader()
test_dataloader = INTELDataset(batch_size=64).get_test_dataloader()

print(f"Train size {train_dataloader.dataset.dataframe.shape}, accuracy {sum(train_dataloader.dataset.dataframe.llm_label == train_dataloader.dataset.dataframe.label) / len(train_dataloader.dataset.dataframe)}")
print(f"Valid size {valid_dataloader.dataset.dataframe.shape}, accuracy {sum(valid_dataloader.dataset.dataframe.llm_label == valid_dataloader.dataset.dataframe.label) / len(valid_dataloader.dataset.dataframe)}")
print(f"Test size {test_dataloader.dataset.dataframe.shape}, accuracy {sum(test_dataloader.dataset.dataframe.llm_label == test_dataloader.dataset.dataframe.label) / len(test_dataloader.dataset.dataframe)}")

vocab, vocab_size = INTELDataset().get_vocab()

# Printing vocab_size to verify
print(f"Vocabulary size, including special tokens: {vocab_size}")



best_valid_loss = float('inf')

# # Model, Loss, and Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = MultiTaskModel(num_classes=6, vocab_size=vocab_size).to(device)
criterion_class = nn.CrossEntropyLoss()
criterion_desc = nn.CrossEntropyLoss()  # For simplicity, adjust for sequence generation
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20  # Specify the number of epochs

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, labels, captions in tqdm.tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        tokenized_captions = [torch.tensor([vocab.get(token, vocab['<unk>']) for token in caption.split()], dtype=torch.long).to(device) for caption in captions]
        caption_indices_tensor = pad_sequence(tokenized_captions, batch_first=True, padding_value=vocab['<pad>'])

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
            images, labels = images.to(device), labels.to(device)
            
            # Prepare validation captions
            tokenized_captions = [torch.tensor([vocab.get(token, vocab['<unk>']) for token in caption.split()], dtype=torch.long).to(device) for caption in captions]
            caption_indices_tensor = pad_sequence(tokenized_captions, batch_first=True, padding_value=vocab['<pad>'])
            
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
        
# # Test the model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
test_loss = 0
correct_preds = 0
total_preds = 0
with torch.no_grad():
    for images, labels, captions in test_dataloader:
        # Prepare test captions
        images, labels = images.to(device), labels.to(device)
        
        tokenized_captions = [torch.tensor([vocab.get(token, vocab['<unk>']) for token in caption.split()], dtype=torch.long).to(device) for caption in captions]
        caption_indices_tensor = pad_sequence(tokenized_captions, batch_first=True, padding_value=vocab['<pad>'])
        
        class_logits, description_logits = model(images, caption_indices_tensor)
        
        loss_class = criterion_class(class_logits, labels)
        loss_desc = criterion_desc(description_logits.view(-1, vocab_size), caption_indices_tensor.view(-1))
        
        loss = loss_class + 0.5 * loss_desc
        test_loss += loss.item()
        
        # Calculate accuracy
        _, predicted_labels = torch.max(class_logits, 1)
        correct_preds += (predicted_labels == labels).sum().item()
        total_preds += labels.size(0)

test_loss /= len(test_dataloader)
test_acc = correct_preds / total_preds
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
