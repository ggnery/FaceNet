from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms
from typing import Optional
import random
from collections import defaultdict
from PIL import Image
from pathlib import Path
from mtcnn import MTCNN
import torch
import numpy as np

class VGGFace2Dataset(Dataset):
    """
    VGGFace2 dataset loader with proper preprocessing for FaceNet.
    """
    
    def __init__(self, root_dir: str, split: str = 'train', 
                 transform: Optional[transforms.Compose] = None):
        """
        Initialize VGGFace2 dataset.
        
        Args:
            root_dir: Path to VGGFace2 dataset
            split: 'train' or 'test'
            transform: Image transformations
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.data_dir = self.root_dir / split
        
        if torch.cuda.is_available():
            self.mtcnn = MTCNN(device='GPU:0')
        else:
            self.mtcnn = MTCNN(device='CPU:0')
        
        # Default FaceNet preprocessing
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)),  # InceptionResNetV2 input size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
            
        # Build dataset
        self.build_dataset()
        
    def build_dataset(self):
        """Build dataset by scanning directory structure."""
        self.samples = []
        self.label_to_indices = defaultdict(list)
        self.labels = []
        
        # Scan all identity folders
        identity_folders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for label_idx, identity_folder in enumerate(identity_folders):
            # Get all images for this identity
            image_files = list(identity_folder.glob('*.jpg')) + list(identity_folder.glob('*.png'))
            
            for img_path in image_files:
                sample_idx = len(self.samples)
                self.samples.append((str(img_path), label_idx))
                self.label_to_indices[label_idx].append(sample_idx)
                self.labels.append(label_idx)
                
        self.num_identities = len(identity_folders)
        print(f"Loaded {len(self.samples)} images from {self.num_identities} identities")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
         
        faces = self.mtcnn.detect_faces(np.array(image, dtype=np.uint8))

        # If no faces detected, use the whole image
        if faces:
            # Use the face with highest confidence
            face = max(faces, key=lambda x: x['confidence'])
            x, y, w, h = face['box']
            
            # Ensure coordinates are within image bounds
            img_width, img_height = image.size
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_width - x)
            h = min(h, img_height - y)
            image = image.crop((x, y, x + w, y + h))
        
        if self.transform:
            image = self.transform(image)

        return image, label


class FaceNetBatchSampler(Sampler):
    """
    Custom batch sampler for FaceNet training.
    Ensures ~40 faces per identity per batch with total batch size ~1800.
    """
    
    def __init__(self, dataset: VGGFace2Dataset, faces_per_identity: int = 40, 
                 num_identities_per_batch: int = 45):
        """
        Initialize the sampler.
        
        Args:
            dataset: VGGFace2Dataset instance
            faces_per_identity: Number of faces per identity in batch (paper: ~40)
            num_identities_per_batch: Number of identities per batch
                                     (45 * 40 = 1800 total batch size)
        """
        self.dataset = dataset
        self.faces_per_identity = faces_per_identity
        self.num_identities_per_batch = num_identities_per_batch
        self.batch_size = faces_per_identity * num_identities_per_batch
        
        # Filter identities that have enough samples
        self.valid_identities = [
            label for label, indices in dataset.label_to_indices.items()
            if len(indices) >= faces_per_identity
        ]
        
        print(f"Found {len(self.valid_identities)} identities with >= {faces_per_identity} samples")
        
        # Cache for efficiency
        self.num_valid_identities = len(self.valid_identities)
        
    def __iter__(self):
        """Generate batches according to FaceNet sampling strategy."""
        # Shuffle identities for each epoch
        identity_order = self.valid_identities.copy()
        random.shuffle(identity_order)
        
        # Generate batches
        for i in range(0, len(identity_order), self.num_identities_per_batch):
            batch_identities = identity_order[i:i + self.num_identities_per_batch]
            
            if len(batch_identities) < self.num_identities_per_batch:
                # For the last incomplete batch, add random identities
                remaining = self.num_identities_per_batch - len(batch_identities)
                extra_identities = random.sample(self.valid_identities, remaining)
                batch_identities.extend(extra_identities)
            
            # Sample faces for each identity
            batch_indices = []
            for identity in batch_identities:
                identity_samples = self.dataset.label_to_indices[identity]
                
                # Sample with replacement if needed
                if len(identity_samples) >= self.faces_per_identity:
                    selected = random.sample(identity_samples, self.faces_per_identity)
                else:
                    selected = random.choices(identity_samples, k=self.faces_per_identity)
                    
                batch_indices.extend(selected)
            
            # Shuffle within batch for better mixing
            random.shuffle(batch_indices)
            yield batch_indices
            
    def __len__(self):
        """Number of batches per epoch."""
        return (self.num_valid_identities + self.num_identities_per_batch - 1) // self.num_identities_per_batch
