import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms
from typing import List, Tuple, Optional, Dict, Any
import glob
from PIL import Image


class DAVIDSimDataset(Dataset):
    """
    Dataset for loading DAVID-sim video sequences with pixel-wise labels.
    """
    # Define the pedestrian color in RGB
    PEDESTRIAN_COLOR = (220, 20, 60)  # RGB format
    
    def __init__(
        self,
        image_root_dir: str,
        label_root_dir: str,
        video_folders: List[str],
        sequence_length: int = 16,
        transform=None,
        target_transform=None,
        stride: int = 1,
        return_metadata: bool = False
    ):
        """
        Args:
            image_root_dir: Root directory containing image folders (DAVID-sim/m1596437/Images)
            label_root_dir: Root directory containing label folders (DAVID-sim/m1596437/Labels)
            video_folders: List of folder names containing frames for each video sequence
            sequence_length: Number of frames to include in each sequence
            transform: Transforms to apply to images
            target_transform: Transforms to apply to labels
            stride: Step size when sampling frames from sequences
            return_metadata: Whether to return metadata about the sequence
        """
        self.image_root_dir = image_root_dir
        self.label_root_dir = label_root_dir
        self.video_folders = video_folders
        self.sequence_length = sequence_length
        self.transform = transform
        self.target_transform = target_transform
        self.stride = stride
        self.return_metadata = return_metadata
        
        # Pre-compute sequence indices and metadata
        self.samples = self._prepare_samples()
        
    def _prepare_samples(self) -> List[Dict[str, Any]]:
        """
        Pre-compute all valid sequences across all video folders.
        """
        samples = []
        
        for video_idx, video_folder in enumerate(self.video_folders):
            image_folder_path = os.path.join(self.image_root_dir, video_folder)
            
            # Get all image files and sort them to ensure correct order
            image_files = sorted(glob.glob(os.path.join(image_folder_path, "*.jpg")) + 
                                glob.glob(os.path.join(image_folder_path, "*.png")))
            
            if not image_files:
                print(f"Warning: No images found in {image_folder_path}")
                continue
                
            # Calculate valid sequences
            num_frames = len(image_files)
            
            for start_frame in range(0, num_frames - self.sequence_length + 1, self.stride):
                samples.append({
                    'video_idx': video_idx,
                    'video_folder': video_folder,
                    'start_frame': start_frame,
                    'frame_indices': list(range(start_frame, start_frame + self.sequence_length)),
                    'sequence_name': f"{video_folder}_{start_frame}"
                })
                
        return samples
    
    def _load_image_sequence(self, video_folder: str, frame_indices: List[int]) -> torch.Tensor:
        """
        Load a sequence of frames from a video folder.
        """
        image_folder_path = os.path.join(self.image_root_dir, video_folder)
        
        # Get all image files and sort them
        image_files = sorted(glob.glob(os.path.join(image_folder_path, "*.jpg")) + 
                            glob.glob(os.path.join(image_folder_path, "*.png")))
        
        frames = []
        for idx in frame_indices:
            if idx < len(image_files):
                # Load image
                image_path = image_files[idx]
                frame = cv2.imread(image_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                # If index out of bounds, use the last available frame or zeros
                if frames:
                    frame = frames[-1].copy()
                else:
                    frame = np.zeros((360, 640, 3), dtype=np.uint8)
                    
            frames.append(frame)
            
        # Stack frames into a single tensor [T, H, W, C]
        sequence = np.stack(frames, axis=0)
        
        # Convert to torch tensor [T, C, H, W]
        sequence = torch.from_numpy(sequence).permute(0, 3, 1, 2).float() / 255.0
        
        return sequence
    
    def _load_label_sequence(self, video_folder: str, frame_indices: List[int]) -> torch.Tensor:
        """
        Load a sequence of label frames and convert to binary pedestrian masks.
        """
        label_folder_path = os.path.join(self.label_root_dir, video_folder)
        
        # Get all label files and sort them
        label_files = sorted(glob.glob(os.path.join(label_folder_path, "*.jpg")) + 
                            glob.glob(os.path.join(label_folder_path, "*.png")))
        
        labels = []
        for idx in frame_indices:
            if idx < len(label_files):
                # Load label image
                label_path = label_files[idx]
                label = cv2.imread(label_path)
                label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
                
                # Create binary mask for pedestrians
                # Check where pixels match the pedestrian color
                pedestrian_mask = np.all(label == self.PEDESTRIAN_COLOR, axis=2).astype(np.uint8)
            else:
                # If index out of bounds, use the last available label or zeros
                if labels:
                    pedestrian_mask = labels[-1].copy()
                else:
                    # Assuming a default shape - adjust as needed
                    pedestrian_mask = np.zeros((360, 640), dtype=np.uint8)
                    
            labels.append(pedestrian_mask)
            
        # Stack labels into a single tensor [T, H, W]
        label_sequence = np.stack(labels, axis=0)
        
        # Convert to torch tensor
        label_sequence = torch.from_numpy(label_sequence).long()
        
        return label_sequence
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        sample = self.samples[idx]
        
        # Load data
        image_sequence = self._load_image_sequence(
            sample['video_folder'], 
            sample['frame_indices']
        )
        
        label_sequence = self._load_label_sequence(
            sample['video_folder'], 
            sample['frame_indices']
        )
        
        # Apply transforms
        if self.transform:
            image_sequence = self.transform(image_sequence)
            
        if self.target_transform:
            label_sequence = self.target_transform(label_sequence)
            
        if self.return_metadata:
            return image_sequence, label_sequence, sample
        
        return image_sequence, label_sequence


class DAVIDSimDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for DAVID-sim dataset.
    """
    def __init__(
        self,
        image_root_dir: str = "DAVID-sim/m1596437/Images",
        label_root_dir: str = "DAVID-sim/m1596437/Labels",
        batch_size: int = 4,
        sequence_length: int = 16,
        stride: int = 8,
        num_workers: int = 4,
        val_split: float = 0.2,
        test_split: float = 0.1,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        target_transform=None,
        seed: int = 42
    ):
        """
        Args:
            image_root_dir: Root directory containing image folders
            label_root_dir: Root directory containing label folders
            batch_size: Batch size for data loaders
            sequence_length: Number of frames per sequence
            stride: Frame stride when creating sequences
            num_workers: Number of workers for data loading
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            train_transform: Transforms for training data
            val_transform: Transforms for validation data
            test_transform: Transforms for test data
            target_transform: Transforms for labels
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.image_root_dir = image_root_dir
        self.label_root_dir = label_root_dir
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.stride = stride
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.train_transform = train_transform
        self.val_transform = val_transform or train_transform
        self.test_transform = test_transform or val_transform
        self.target_transform = target_transform
        self.seed = seed
        
        self.train_folders = []
        self.val_folders = []
        self.test_folders = []
        
    def _get_video_folders(self) -> List[str]:
        """
        Get all video folders in the image root directory.
        """
        # Get all subdirectories in the image root directory
        return [d for d in os.listdir(self.image_root_dir) 
                if os.path.isdir(os.path.join(self.image_root_dir, d))]
    
    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for training, validation, and testing.
        """
        video_folders = self._get_video_folders()
        print(f"Found {len(video_folders)} video folders")
        
        # Shuffle with fixed seed for reproducibility
        np.random.seed(self.seed)
        np.random.shuffle(video_folders)
        
        # Split data
        num_videos = len(video_folders)
        num_val = max(1, int(num_videos * self.val_split))
        num_test = max(1, int(num_videos * self.test_split))
        num_train = num_videos - num_val - num_test
        
        # Assign folders to splits
        self.train_folders = video_folders[:num_train]
        self.val_folders = video_folders[num_train:num_train+num_val]
        self.test_folders = video_folders[num_train+num_val:]
        
        print(f"Data split: {len(self.train_folders)} train, {len(self.val_folders)} val, {len(self.test_folders)} test")
        
    def train_dataloader(self):
        train_dataset = DAVIDSimDataset(
            image_root_dir=self.image_root_dir,
            label_root_dir=self.label_root_dir,
            video_folders=self.train_folders,
            sequence_length=self.sequence_length,
            transform=self.train_transform,
            target_transform=self.target_transform,
            stride=self.stride
        )
        
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self):
        val_dataset = DAVIDSimDataset(
            image_root_dir=self.image_root_dir,
            label_root_dir=self.label_root_dir,
            video_folders=self.val_folders,
            sequence_length=self.sequence_length,
            transform=self.val_transform,
            target_transform=self.target_transform,
            stride=self.stride * 2  # Wider stride for validation
        )
        
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )
    
    def test_dataloader(self):
        test_dataset = DAVIDSimDataset(
            image_root_dir=self.image_root_dir,
            label_root_dir=self.label_root_dir,
            video_folders=self.test_folders,
            sequence_length=self.sequence_length,
            transform=self.test_transform,
            target_transform=self.target_transform,
            stride=self.stride * 2,
            return_metadata=True
        )
        
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )


# Example transformations
def get_transforms(img_size=(224, 224)):
    """
    Create basic transformations for image sequences.
    """
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        # Resize individual frames
        lambda x: torch.nn.functional.interpolate(x, size=img_size, mode='bilinear', align_corners=False),
        # Add random horizontal flip with 50% probability
        lambda x: torch.flip(x, dims=[3]) if np.random.random() < 0.5 else x,
        # Normalize using ImageNet statistics
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/test transforms (no augmentation)
    val_transform = transforms.Compose([
        lambda x: torch.nn.functional.interpolate(x, size=img_size, mode='bilinear', align_corners=False),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Label transforms - resize to match image size
    target_transform = transforms.Compose([
        lambda x: torch.nn.functional.interpolate(x.unsqueeze(1).float(), size=img_size, mode='nearest').squeeze(1).long()
    ])
    
    return train_transform, val_transform, target_transform


# Example usage
def example_usage():
    # Define transforms
    train_transform, val_transform, target_transform = get_transforms(img_size=(224, 224))
    
    # Create data module
    data_module = DAVIDSimDataModule(
        image_root_dir="../DAVID-sim/m1596437/Images",
        label_root_dir="../DAVID-sim/m1596437/Labels",
        batch_size=4,
        sequence_length=16,
        stride=8,
        train_transform=train_transform,
        val_transform=val_transform,
        target_transform=target_transform,
        num_workers=4
    )
    
    # Set up data
    data_module.setup()
    
    # Get data loaders
    train_loader = data_module.train_dataloader()
    
    # Example iteration
    for batch_idx, (images, labels) in enumerate(train_loader):
        # images shape: [batch_size, sequence_length, channels, height, width]
        # labels shape: [batch_size, sequence_length, height, width]
        print(f"Batch {batch_idx}:")
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break


if __name__ == "__main__":
    example_usage()