from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

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
    
