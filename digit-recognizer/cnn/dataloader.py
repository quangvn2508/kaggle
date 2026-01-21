from torch.utils.data import DataLoader
from dataset import DigitCSVDataset
from torch.utils.data import random_split

train_dataset = DigitCSVDataset("./data/train.csv", has_labels=True, train=True)
val_dataset   = DigitCSVDataset("./data/train.csv", has_labels=True, train=False)

train_size = int(0.85 * len(train_dataset))
val_size   = len(train_dataset) - train_size

train_dataset, _ = random_split(train_dataset, [train_size, val_size])
_, val_dataset   = random_split(val_dataset,   [train_size, val_size])


test_dataset = DigitCSVDataset("./data/test.csv", has_labels=False, train=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset,   batch_size=1000, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
