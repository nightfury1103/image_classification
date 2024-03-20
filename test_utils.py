import torch
from metrics import criterion_class, criterion_desc
from torch.nn.utils.rnn import pad_sequence

def get_model_performance(alpha, model, path, test_dataloader, vocab, vocab_size,device='cuda', is_caption=True):

    # # Test the model
    model.load_state_dict(torch.load(path))
    model.eval()
    test_loss = 0
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for images, labels, captions in test_dataloader:
            # Prepare test captions
            images, labels = images.to(device), labels.to(device)
            
            if is_caption:
                tokenized_captions = [torch.tensor([vocab.get(token, vocab['<unk>']) for token in caption.split()], dtype=torch.long).to(device) for caption in captions]
                caption_indices_tensor = pad_sequence(tokenized_captions, batch_first=True, padding_value=vocab['<pad>'])
                class_logits, description_logits = model(images)
            
                loss_class = criterion_class(class_logits, labels)
                description_logits = description_logits[:, :caption_indices_tensor.size(1), :]

                generated_seq_length = description_logits.size(1)  # Current fixed length (20)
                target_seq_length = caption_indices_tensor.size(1)  # Target sequence length based on the batch

                                # Check if we need to pad or trim the generated descriptions to match the target length
                if generated_seq_length < target_seq_length:
                    # Calculate padding needed
                    pad_length = target_seq_length - generated_seq_length
                    # Pad the description_logits along the sequence dimension (dim=1)
                    padding = torch.zeros(description_logits.size(0), pad_length, description_logits.size(2)).to(description_logits.device)
                    description_logits_padded = torch.cat([description_logits, padding], dim=1)
                elif generated_seq_length > target_seq_length:
                    # Trim the description_logits to match the target sequence length
                    description_logits_padded = description_logits[:, :target_seq_length, :]
                else:
                    # Lengths match, no need to adjust
                    description_logits_padded = description_logits

                loss_desc = criterion_desc(description_logits_padded.reshape(-1, vocab_size), caption_indices_tensor.view(-1))

                loss = alpha * loss_class + (1 - alpha) * loss_desc
            else:
                class_logits = model(images)
                loss = criterion_class(class_logits, labels)
            
            test_loss += loss.item()
            
            # Calculate accuracy
            _, predicted_labels = torch.max(class_logits, 1)
            correct_preds += (predicted_labels == labels).sum().item()
            total_preds += labels.size(0)

    test_loss /= len(test_dataloader)
    test_acc = correct_preds / total_preds
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')