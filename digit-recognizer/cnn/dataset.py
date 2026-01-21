import random
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class DigitCSVDataset(Dataset):
    def __init__(self, csv_file, has_labels=True, train=False):
        self.data = pd.read_csv(csv_file)
        self.has_labels = has_labels
        self.train = train

        if has_labels:
            self.labels = self.data.iloc[:, 0].values
            self.images = self.data.iloc[:, 1:].values
        else:
            self.images = self.data.values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28).astype(np.float32)
        image = torch.tensor(image / 255.0).unsqueeze(0)  # [1,28,28]
        # image = (image - 0.1307) / 0.3081

        if self.train:
            image = self.augment(image)

        if self.has_labels:
            label = torch.tensor(self.labels[idx]).long()
            return image, label
        else:
            return image
    
    def augment(self, image):
        # small random shift
        if random.random() < 0.5:
            shift = random.randint(-2, 2)
            image = torch.roll(image, shifts=shift, dims=1)

        # random noise
        if random.random() < 0.3:
            image = image + 0.02 * torch.randn_like(image)

        return torch.clamp(image, 0.0, 1.0)
