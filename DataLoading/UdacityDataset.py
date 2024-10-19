import json
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os
from skimage import io
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class UdacityDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def decode_image(self, img_str):
        image = Image.open(BytesIO(base64.b64decode(img_str)))
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)  # Per image transformation
        return image

    def read_data_single(self, idx):
        """Reads data for a single index."""
        keys = list(self.dataset.keys())
        frame_key = keys[idx]
        entry = self.dataset[frame_key]

        images = {cam: self.decode_image(entry['images'][cam]) for cam in ['left', 'center', 'right']}

        data = {
            'images': images,  # Dictionary of tensors
            'timestamp': entry['timestamp'],
            'angle': torch.tensor(entry['angle'], dtype=torch.float),
            'torque': torch.tensor(entry['torque'], dtype=torch.float),
            'speed': torch.tensor(entry['speed'], dtype=torch.float)
        }
        return data

    def split_dataset(self, test_size=0.2, random_state=None):
        keys = list(self.dataset.keys())
        train_keys, valid_keys = train_test_split(keys, test_size=test_size, random_state=random_state)

        train_dataset = UdacityDataset(dataset=train_keys, transform=self.transform)
        valid_dataset = UdacityDataset(dataset=valid_keys, transform=self.transform)

        return train_dataset, valid_dataset

    def read_data(self, idx):
        if isinstance(idx, list):
            batch_data = []
            for i in tqdm(idx, desc="Loading data", unit="batch"):
                batch_data.append(self.read_data_single(i))
            images = {cam: torch.stack([data['images'][cam] for data in batch_data]) for cam in
                      ['left', 'center', 'right']}
            batch = {
                'images': images,
                'timestamp': [data['timestamp'] for data in batch_data],
                'angle': torch.stack([data['angle'] for data in batch_data]),
                'torque': torch.stack([data['torque'] for data in batch_data]),
                'speed': torch.stack([data['speed'] for data in batch_data])
            }
            return batch
        else:
            return self.read_data_single(idx)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.read_data(idx)

