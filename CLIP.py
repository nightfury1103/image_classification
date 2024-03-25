import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from data_utils import INTELDataset
import torch
from tqdm import tqdm

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move the model to the selected device
model.to(device)

batch_size = 16

train_loader = INTELDataset(batch_size=batch_size).get_train_dataloader()
epochs = range(1)

for image, label, caption in train_loader:
    image = Image.open(image[0])
    inputs = processor(image, label[0], return_tensors="pt")
    # inputs['labels'] = processor(text=caption[0], return_tensors="pt")["input_ids"]

    outputs = model(**inputs)
    break
print(outputs.loss)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
for epoch in epochs:
    for i, (image_paths, labels, captions) in enumerate(tqdm(train_loader)):
        batch_images = []
        # Process each image in the batch
        for image_path in image_paths:
            with Image.open(image_path) as img:
                img = img.convert("RGB")  # Ensure image is in RGB
                batch_images.append(img)
                
        inputs = processor(batch_images, labels, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        loss = outputs.loss
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")
print(outputs.loss)

# Define your save path
model_save_path = 'classification.pt'

# Save the model state dict
torch.save(model.state_dict(), model_save_path)

