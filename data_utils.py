import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

curr_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_dir)
sys.path.append(os.path.join(curr_dir, '../'))

from config import DATA_DIR

class ImageDataset(Dataset):
    def __init__(self, dataframe, indices=None):
        if indices:
            self.dataframe = dataframe.iloc[indices].reset_index(drop=True)
        else:
            self.dataframe = dataframe
        self.transform = Compose([
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
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
    
class INTELDataset:
    def __init__(self, batch_size=32):
        self.train_df = pd.read_csv(DATA_DIR['INTEL']['train'], index_col=0)
        self.test_df = pd.read_csv(DATA_DIR['INTEL']['test'], index_col=0)

        train_idx, valid_idx = train_test_split(range(len(self.train_df)), test_size=0.2, random_state=42)      
        
        self.train = ImageDataset(self.train_df, train_idx)
        self.valid = ImageDataset(self.train_df, valid_idx)
        self.test = ImageDataset(self.test_df)
        
        self.batch_size = batch_size
        
    def get_train_dataloader(self, shuffle=True):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=shuffle)

    def get_valid_dataloader(self, shuffle=False):
        return DataLoader(self.valid, batch_size=self.batch_size, shuffle=shuffle)

    def get_test_dataloader(self, shuffle=False):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=shuffle)
    
    def get_vocab(self):
        # Initialize an empty set to collect unique tokens
        unique_tokens = set()

        # Iterate over the captions in your dataset to collect all unique tokens
        for caption in self.train_df['description']:
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

        return vocab, vocab_size
    
class CIFARDataset:
    def __init__(self) -> None:
        dataset = load_dataset("cifar10")
        self.train
        
        
# CIFARDataset()