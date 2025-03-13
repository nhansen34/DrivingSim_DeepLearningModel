""" Notes from Jake
PedestrianDataset class and your load_datasets class should be in a python script 
in your dataset folder, so you don't need to rewrite it every time you have a new
notebooks.
It's also a good idea to use the path to your data folder as an argument instead of 
hard coding /projects/dsci410_510/ into your csv, in case you ever want to run this 
code somewhere else.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import cv2

class PedestrianDataset(Dataset):
    def __init__(self, dataframe, transform=True):
        self.dataframe = dataframe
        self.transform = transform
        self.label_mapping = {"Low": 0, "Medium": 1, "High": 2}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["unlabeled_image_path"]
        label = self.label_mapping[self.dataframe.iloc[idx]["risk_level"]]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)

def load_datasets(csv_path, batch_size=32):
    df = pd.read_csv(csv_path).sample(frac=1).reset_index(drop=True)
    df = df.drop(["labeled_image_path", "pedestrian_pixels"], axis=1)
    
    train_df = df[:int(0.8 * len(df))]
    val_df = df[int(0.8 * len(df)):int(0.9 * len(df))]
    test_df = df[int(0.9 * len(df)):]

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = PedestrianDataset(train_df, transform=transform)
    val_dataset = PedestrianDataset(val_df, transform=transform)
    test_dataset = PedestrianDataset(test_df, transform=transform)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    )