from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes, vocab, vocab_size, embed_size=256, hidden_size=512, num_layers=1):
        super(MultiTaskModel, self).__init__()
        self.num_layers = num_layers 
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

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
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self.feature_transform = nn.Linear(in_features, embed_size)

        # Your description head remains the same
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.description_head = nn.Linear(hidden_size, vocab_size)

        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        self.start_token_index = vocab['<sos>']

    def forward(self, images):
        features = self.backbone(images)
        class_logits = self.classification_head(features)
        
        # Prepare features for the description generation
        features = features.unsqueeze(1)  # Add sequence dimension
        features = self.feature_transform(features)  # Transform features
        
        h, c = self.init_hidden_state(features.size(0))  # Initialize hidden state
        
        # Start token for generation (assuming you have a start token defined)
        start_tokens = torch.full((features.size(0), 1), fill_value=self.start_token_index, device=features.device, dtype=torch.long)
        embeddings = self.word_embeddings(start_tokens)  # Get embeddings for start tokens
        
        descriptions = []
        for _ in range(20):  # Sequence length
            lstm_out, (h, c) = self.lstm(embeddings, (h, c))
            outputs = self.description_head(lstm_out.squeeze(1))
            descriptions.append(outputs)
            _, predicted = outputs.max(1)  # Get the most likely next word index
            
            # Prepare embeddings for the next step based on predicted indices
            embeddings = self.word_embeddings(predicted).unsqueeze(1)
        
        descriptions = torch.stack(descriptions, 1)
        
        return class_logits, descriptions

    def init_hidden_state(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
    
