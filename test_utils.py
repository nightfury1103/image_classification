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
                
                class_logits, description_logits = model(images, caption_indices_tensor)
            
                loss_class = criterion_class(class_logits, labels)
                loss_desc = criterion_desc(description_logits.view(-1, vocab_size), caption_indices_tensor.view(-1))
            
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