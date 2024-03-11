import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import pandas as pd

curr_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_dir)
sys.path.append(os.path.join(curr_dir, '../'))

from config import DATA_DIR

class ImageDataset(Dataset):
    def __init__(self, dataframe, indices, transform=None):
        self.dataframe = dataframe.iloc[indices].reset_index(drop=True)
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
    def __init__(self):
        self.train = pd.read_csv(DATA_DIR['INTEL']['train'], index_col=0)
        self.valid = self.train.sample(frac=0.7, random_state=42)
        self.test = self.train.drop(self.valid.index)
        # self.test = pd.read_csv(DATA_DIR['test'], index_col=0)
        
    def get_train_dataloader(self, batch_size=32, shuffle=True):
        return DataLoader(self.train, batch_size=batch_size, shuffle=shuffle)

    def get_valid_dataloader(self, batch_size=32, shuffle=False):
        return DataLoader(self.valid, batch_size=batch_size, shuffle=shuffle)

    def get_test_dataloader(self, batch_size=32, shuffle=False):
        return DataLoader(self.test, batch_size=batch_size, shuffle=shuffle)
    