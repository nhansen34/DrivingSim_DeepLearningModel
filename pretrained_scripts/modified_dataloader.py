import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os

class PedestrianDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.label_mapping = {"Low": 0, "Medium": 1, "High": 2}
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["unlabeled_image_path"]
        label = self.label_mapping[self.dataframe.iloc[idx]["risk_level"]]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            # Create a blank image as fallback
            image = np.zeros((128, 128, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label)

def load_datasets(csv_path, batch_size=32, use_pretrained=True):
    df = pd.read_csv(csv_path).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Remove unnecessary columns
    if "labeled_image_path" in df.columns and "pedestrian_pixels" in df.columns:
        df = df.drop(["labeled_image_path", "pedestrian_pixels"], axis=1)
    
    # Split data
    train_df = df[:int(0.8 * len(df))]
    val_df = df[int(0.8 * len(df)):int(0.9 * len(df))]
    test_df = df[int(0.9 * len(df)):]
    
    # Different transforms for pretrained models vs custom models
    if use_pretrained:
        # ImageNet normalization for pretrained models
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Standard input size for many pretrained models
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ])
        
        # Optional: add data augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        # Original normalization for custom model
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Simple data augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize((140, 140)),
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    # Create datasets with appropriate transforms
    train_dataset = PedestrianDataset(train_df, transform=train_transform)
    val_dataset = PedestrianDataset(val_df, transform=transform)
    test_dataset = PedestrianDataset(test_df, transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader
