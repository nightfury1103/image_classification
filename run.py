import torch
import argparse

from data_utils import INTELDataset, CIFARDataset
from model_utils import MultiTaskModel
from test_utils import get_model_performance
from train_utils import trainer


def run(args):
    if args.dataset == 'INTEL':
        train_dataloader = INTELDataset(batch_size=args.batch_size).get_train_dataloader()
        valid_dataloader = INTELDataset(batch_size=args.batch_size).get_valid_dataloader()
        test_dataloader = INTELDataset(batch_size=args.batch_size).get_test_dataloader()
        vocab, vocab_size = INTELDataset().get_vocab()
        num_classes = INTELDataset().num_classes

    elif args.dataset == 'CIFAR':
        train_dataloader = CIFARDataset(batch_size=args.batch_size).get_train_dataloader()
        valid_dataloader = CIFARDataset(batch_size=args.batch_size).get_valid_dataloader()
        test_dataloader = CIFARDataset(batch_size=args.batch_size).get_test_dataloader()
        vocab, vocab_size = CIFARDataset().get_vocab()
        num_classes = CIFARDataset().num_classes

    print(f"Train size {train_dataloader.dataset.dataframe.shape}, accuracy {sum(train_dataloader.dataset.dataframe.llm_label == train_dataloader.dataset.dataframe.label) / len(train_dataloader.dataset.dataframe)}")
    print(f"Valid size {valid_dataloader.dataset.dataframe.shape}, accuracy {sum(valid_dataloader.dataset.dataframe.llm_label == valid_dataloader.dataset.dataframe.label) / len(valid_dataloader.dataset.dataframe)}")
    print(f"Test size {test_dataloader.dataset.dataframe.shape}, accuracy {sum(test_dataloader.dataset.dataframe.llm_label == test_dataloader.dataset.dataframe.label) / len(test_dataloader.dataset.dataframe)}")


    # Printing vocab_size to verify
    print(f"Vocabulary size, including special tokens: {vocab_size}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = MultiTaskModel(num_classes=num_classes, vocab_size=vocab_size).to(device)
    
    print('---MODE CAPTION---: ', args.is_caption)

    trainer(num_epochs=args.epochs, alpha=args.alpha, model=model, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, device=device, vocab=vocab, vocab_size=vocab_size, is_caption=args.is_caption)

    get_model_performance(model=model, alpha=args.alpha, path='models/best_model_loss.pth', test_dataloader=test_dataloader, vocab=vocab, vocab_size=vocab_size, device=device, is_caption=args.is_caption)
    get_model_performance(model=model, alpha=args.alpha, path='models/best_model_acc.pth', test_dataloader=test_dataloader, vocab=vocab, vocab_size=vocab_size, device=device, is_caption=args.is_caption)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--is_caption', type=bool, default=True)
    parser.add_argument('--alpha', type=float, default=0.5)

    args = parser.parse_args()

    # dic = {'dataset': 'INTEL', 'batch_size': 64, 'epochs': 1, 'is_caption': True, 'alpha': 0.5}
    # from types import SimpleNamespace
    # args = SimpleNamespace(**dic)

    run(args)