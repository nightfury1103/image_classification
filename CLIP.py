import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from data_utils import INTELDataset
from tqdm import tqdm

# Initialize the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Check for GPU availability and move the model to the selected device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

batch_size = 8
train_loader = INTELDataset(batch_size=batch_size).get_train_dataloader()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()

for epoch in range(1):
    for i, (image_paths, labels, captions) in enumerate(tqdm(train_loader)):
        batch_images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            batch_images.append(image)
            
        # Process images and captions and then move processed data to the correct device
        inputs = processor(images=batch_images, text=captions, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")
            
# Save the model's state_dict
torch.save(model.state_dict(), 'classification.pt')
