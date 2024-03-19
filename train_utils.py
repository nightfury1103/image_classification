import os, sys
import torch

import torch
import tqdm
from torch.nn.utils.rnn import pad_sequence

curr_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_dir)
sys.path.append(os.path.join(curr_dir, '../'))

from metrics import criterion_class, criterion_desc


def trainer(num_epochs, alpha, model, train_dataloader, valid_dataloader, vocab, vocab_size, device='cuda', is_caption=True):
    best_valid_loss = float('inf')
    best_valid_acc = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels, captions in tqdm.tqdm(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if is_caption:
                # Prepare training captions
                tokenized_captions = [torch.tensor([vocab.get(token, vocab['<unk>']) for token in caption.split()], dtype=torch.long).to(device) for caption in captions]
                caption_indices_tensor = pad_sequence(tokenized_captions, batch_first=True, padding_value=vocab['<pad>'])

                class_logits, description_logits = model(images, caption_indices_tensor)

                loss_class = criterion_class(class_logits, labels)
                loss_desc = criterion_desc(description_logits.view(-1, vocab_size), caption_indices_tensor.view(-1))

                loss = alpha * loss_class + (1 - alpha) * loss_desc
            else:
                class_logits = model(images)
                loss_class = criterion_class(class_logits, labels)
                loss = loss_class

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

                if is_caption:
                    # Prepare validation captions
                    tokenized_captions = [torch.tensor([vocab.get(token, vocab['<unk>']) for token in caption.split()], dtype=torch.long).to(device) for caption in captions]
                    caption_indices_tensor = pad_sequence(tokenized_captions, batch_first=True, padding_value=vocab['<pad>'])

                    class_logits, description_logits = model(images, caption_indices_tensor)

                    loss_class = criterion_class(class_logits, labels)
                    loss_desc = criterion_desc(description_logits.view(-1, vocab_size), caption_indices_tensor.view(-1))

                    loss = alpha * loss_class + (1 - alpha) * loss_desc
                else:
                    class_logits = model(images)
                    loss_class = criterion_class(class_logits, labels)
                    loss = loss_class

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
            torch.save(model.state_dict(), 'models/best_model_loss.pth')

        # Save model if it has the best validation accuracy
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), 'models/best_model_acc.pth')

    # Save the last epoch
    torch.save(model.state_dict(), 'models/last_model.pth')

