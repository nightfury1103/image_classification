import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from data_utils import INTELDataset
from tqdm import tqdm
import numpy as np
import random
from metrics import tokenize_and_pad_texts

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Ensure deterministic behavior in CuDNN (if using CUDA)
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Initialize the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Check for GPU availability and move the model to the selected device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

batch_size = 4
train_loader = INTELDataset(batch_size=batch_size).get_train_dataloader()
valid_loader = INTELDataset(batch_size=batch_size).get_valid_dataloader()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()

alpha = 0.5

train_loss = 0
for epoch in range(5):
    for i, (image_paths, labels, captions) in enumerate(tqdm(train_loader)):
        batch_images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            batch_images.append(image)
            
        # Classification task
        inputs = processor(images=batch_images, text=['predict: '] * len(labels), return_tensors="pt", padding=True, truncation=True)
        encode_labels = processor(images=batch_images, text=list(labels), return_tensors="pt", padding=True, truncation=True)

        token_1 = inputs['input_ids']
        token_2 = encode_labels['input_ids']
        token_1, token_2 = tokenize_and_pad_texts(token_1, token_2)
        
        inputs["input_ids"] = token_1
        inputs["labels"] = token_2
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        classification_output = model(**inputs)
        classification_loss = classification_output.loss
        
        # Captioning task
        inputs = processor(images=batch_images, text=['describe: '] * len(captions), return_tensors="pt", padding=True, truncation=True, max_length=256)
        encode_captions = processor(images=batch_images, text=list(captions), return_tensors="pt", padding=True, truncation=True)

        token_1 = inputs['input_ids']
        token_2 = encode_captions['input_ids']
        token_1, token_2 = tokenize_and_pad_texts(token_1, token_2)

        inputs["input_ids"] = token_1
        inputs["labels"] = token_2
        
        sequence_length_diff = inputs["input_ids"].shape[1] - inputs["attention_mask"].shape[1]
        if sequence_length_diff > 0:
            # Apply padding to the sequence length dimension (the last dimension)
            attention_mask_padded = torch.nn.functional.pad(inputs['attention_mask'], (0, sequence_length_diff), value=0)
        
        inputs['attention_mask'] = attention_mask_padded

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        captioning_output = model(**inputs)
        captioning_loss = captioning_output.loss

        loss = alpha * classification_loss + (1 - alpha) * captioning_loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")
        train_loss += loss.item()

    # Validation
    model.eval()
    valid_loss = 0
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for i, (image_paths, labels, captions) in enumerate(tqdm(valid_loader)):
            batch_images = []
            for image_path in image_paths:
                image = Image.open(image_path).convert("RGB")
                batch_images.append(image)

            # Classification task
            inputs = processor(images=batch_images, text=['predict: '] * len(labels), return_tensors="pt", padding=True, truncation=True)
            encode_labels = processor(images=batch_images, text=list(labels), return_tensors="pt", padding=True, truncation=True)

            token_1 = inputs['input_ids']
            token_2 = encode_labels['input_ids']
            token_1, token_2 = tokenize_and_pad_texts(token_1, token_2)
            
            inputs["input_ids"] = token_1
            inputs["labels"] = token_2
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            classification_output = model(**inputs)
            classification_loss = classification_output.loss

            # Captioning task
            inputs = processor(images=batch_images, text=['describe: '] * len(captions), return_tensors="pt", padding=True, truncation=True, max_length=256)
            encode_captions = processor(images=batch_images, text=list(captions), return_tensors="pt", padding=True, truncation=True)

            token_1 = inputs['input_ids']
            token_2 = encode_captions['input_ids']
            token_1, token_2 = tokenize_and_pad_texts(token_1, token_2)

            inputs["input_ids"] = token_1
            inputs["labels"] = token_2
            
            sequence_length_diff = inputs["input_ids"].shape[1] - inputs["attention_mask"].shape[1]
            if sequence_length_diff > 0:
                # Apply padding to the sequence length dimension (the last dimension)
                attention_mask_padded = torch.nn.functional.pad(inputs['attention_mask'], (0, sequence_length_diff), value=0)
            
            inputs['attention_mask'] = attention_mask_padded

            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            captioning_output = model(**inputs)
            captioning_loss = captioning_output.loss

            loss = alpha * classification_loss + (1 - alpha) * captioning_loss

            valid_loss += loss.item()
            # Calculate accuracy
            predicted_labels = processor.batch_decode(torch.argmax(classification_output.decoder_logits, dim=-1), skip_special_tokens=True)
            correct_preds += (np.array(predicted_labels) == np.array(labels)).sum().item()
            total_preds += len(labels)

        # Average losses
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        valid_acc = correct_preds / total_preds

        print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, Validation Acc: {valid_acc:.4f}')


# Save the model's state_dict
torch.save(model.state_dict(), 'classify.pt')
