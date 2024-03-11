import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
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


class ImageDataset(Dataset):
    def __init__(self, dataframe, indices, transform=None):
        self.dataframe = dataframe.iloc[indices].reset_index(drop=True)
        self.transform = transform
        label_encoder = LabelEncoder()
        self.dataframe['label_encoded'] = label_encoder.fit_transform(self.dataframe['llm_label'])
        self.dataframe['truth_label_encoded'] = label_encoder.transform(self.dataframe['label'])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe['image'][idx]  # Assuming the first column is the image path
        image = Image.open(img_path)
        label = torch.tensor(self.dataframe['truth_label_encoded'][idx])  # Assuming the second column is the label
        description = self.dataframe['description'][idx]  # Assuming the third column is the description

        if self.transform:
            image = self.transform(image)

        return image, label, description

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

df = pd.read_csv('/home/huy/Desktop/HCMUS/image_classification/datasets/INTEL/seg_train/seg_train/trained.csv', index_col=0)

train_idx, temp_idx = train_test_split(range(len(df)), test_size=0.3, random_state=42)
valid_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

train_dataset = ImageDataset(df, train_idx, transform=transform)
valid_dataset = ImageDataset(df, valid_idx, transform=transform)
test_dataset = ImageDataset(df, test_idx, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize an empty set to collect unique tokens
unique_tokens = set()

# Iterate over the captions in your dataset to collect all unique tokens
for caption in df['description']:
    # Simple tokenization by splitting on spaces
    tokens = caption.lower().split()
    unique_tokens.update(tokens)

# Now that we have all unique tokens, let's create a vocabulary dictionary
# that maps each token to a unique index.
vocab = {token: idx for idx, token in enumerate(unique_tokens)}

# Add special tokens to the vocabulary
special_tokens = ['<sos>', '<eos>', '<pad>', '<unk>']
for token in special_tokens:
    if token not in vocab:
        vocab[token] = len(vocab)

# Now, the vocab_size is simply the number of items in the vocabulary dictionary
vocab_size = len(vocab)

# Printing vocab_size to verify
print(f"Vocabulary size, including special tokens: {vocab_size}")

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes, vocab_size):
        super(MultiTaskModel, self).__init__()
        # Load the pre-trained ResNet model
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Instead of completely removing the original fc layer, store its in_features
        in_features = self.backbone.fc.in_features
        
        # Now safely replace the fc layer with nn.Identity
        self.backbone.fc = nn.Identity()
        
        # Use the stored in_features for the subsequent custom layers
        self.classification_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
        # Your description head remains the same
        self.embedding = nn.Embedding(vocab_size, 256)
        self.lstm = nn.LSTM(256, 512, batch_first=True)
        self.description_head = nn.Linear(512, vocab_size)
        
    def forward(self, images, captions=None):
        features = self.backbone(images)  # Shared features
        # Classification
        class_logits = self.classification_head(features)
        
        # Description (Captioning)
        if captions is not None:
            embeddings = self.embedding(captions)
            lstm_out, _ = self.lstm(embeddings)
            description_logits = self.description_head(lstm_out)
            return class_logits, description_logits
        
        return class_logits

best_valid_loss = float('inf')

# # Model, Loss, and Optimizer
model = MultiTaskModel(num_classes=6, vocab_size=vocab_size)
criterion_class = nn.CrossEntropyLoss()
criterion_desc = nn.CrossEntropyLoss()  # For simplicity, adjust for sequence generation
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# num_epochs = 10  # Specify the number of epochs

# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0
#     for images, labels, captions in tqdm.tqdm(train_dataloader):
#         optimizer.zero_grad()

#         # Prepare captions
#         tokens = captions[0].split()  # Assuming a batch size of 1 for simplicity
#         token_indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
#         caption_indices_tensor = torch.tensor(token_indices, dtype=torch.long).unsqueeze(0)
        
#         class_logits, description_logits = model(images, caption_indices_tensor)
        
#         loss_class = criterion_class(class_logits, labels)
#         loss_desc = criterion_desc(description_logits.view(-1, vocab_size), caption_indices_tensor.view(-1))
        
#         loss = loss_class + 0.5 * loss_desc
#         loss.backward()
#         optimizer.step()
        
#         train_loss += loss.item()

#     # Validation step
#     model.eval()
#     valid_loss = 0
#     correct_preds = 0
#     total_preds = 0
#     with torch.no_grad():
#         for images, labels, captions in valid_dataloader:
#             # Prepare validation captions
#             tokens = captions[0].split()  # Assuming a batch size of 1 for simplicity
#             token_indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
#             caption_indices_tensor = torch.tensor(token_indices, dtype=torch.long).unsqueeze(0)
            
#             class_logits, description_logits = model(images, caption_indices_tensor)
            
#             loss_class = criterion_class(class_logits, labels)
#             loss_desc = criterion_desc(description_logits.view(-1, vocab_size), caption_indices_tensor.view(-1))
            
#             loss = loss_class + 0.5 * loss_desc
#             valid_loss += loss.item()
            
#             # Calculate accuracy
#             _, predicted_labels = torch.max(class_logits, 1)
#             correct_preds += (predicted_labels == labels).sum().item()
#             total_preds += labels.size(0)

#     # Average losses
#     train_loss /= len(train_dataloader)
#     valid_loss /= len(valid_dataloader)
#     valid_acc = correct_preds / total_preds
    
#     print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, Validation Acc: {valid_acc:.4f}')

#     # Save model if it has the best validation loss
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), 'best_model.pth')
        
# # Test the model
model.load_state_dict(torch.load('/home/huy/Desktop/HCMUS/image_classification/models/best_model.pth'))
model.eval()
test_loss = 0
correct_preds = 0
total_preds = 0
with torch.no_grad():
    for images, labels, captions in test_dataloader:
        # Prepare test captions
        tokens = captions[0].split()  # Assuming a batch size of 1 for simplicity
        token_indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
        caption_indices_tensor = torch.tensor(token_indices, dtype=torch.long).unsqueeze(0)
        
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
