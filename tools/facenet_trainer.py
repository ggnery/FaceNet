import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import logging
import json
from pathlib import Path
from dataset import VGGFace2Dataset, FaceNetBatchSampler
from model import FaceNetInceptionResNetV2

class FaceNetTrainer:
    """
    Trainer class for FaceNet model following paper specifications.
    """
    
    def __init__(self, model: FaceNetInceptionResNetV2, device: torch.device, 
                 checkpoint_dir: str = './checkpoints'):
        """
        Initialize trainer.
        
        Args:
            model: FaceNet model
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'mining_stats': []
        }
        
    def train(self, train_dataset: VGGFace2Dataset, 
              val_dataset: Optional[VGGFace2Dataset] = None,
              num_epochs: int = 100,
              learning_rate: float = 0.05,
              lr_schedule: Optional[Dict[int, float]] = None,
              faces_per_identity: int = 40,
              num_identities_per_batch: int = 45):
        """
        Train the FaceNet model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate (paper starts with 0.05)
            lr_schedule: Dictionary mapping epoch to learning rate
            faces_per_identity: Faces per identity per batch
            num_identities_per_batch: Number of identities per batch
        """
        # Create custom batch sampler
        batch_sampler = FaceNetBatchSampler(
            train_dataset, 
            faces_per_identity=faces_per_identity,
            num_identities_per_batch=num_identities_per_batch
        )
        
        # Create data loader with custom sampler
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=8,
            pin_memory=True
        )
        
        # Optimizer (paper uses AdaGrad)
        optimizer = optim.Adagrad(self.model.parameters(), lr=learning_rate)
        
        # Learning rate schedule
        if lr_schedule is None:
            # Default schedule from paper
            lr_schedule = {
                0: 0.05,
                500: 0.01,  # After 500 epochs (~500 epochs)
                1000: 0.001
            }
        
        # Training loop
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Batch size: {batch_sampler.batch_size} "
                        f"({faces_per_identity} faces x {num_identities_per_batch} identities)")
        
        for epoch in range(num_epochs):
            # Adjust learning rate
            current_lr = self.adjust_learning_rate(optimizer, epoch, lr_schedule)
            
            # Train one epoch
            train_loss, train_stats = self.train_epoch(train_loader, optimizer, epoch)
            
            # Validation
            val_loss = None
            if val_dataset:
                val_loss = self.validate(val_dataset)
                
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"LR: {current_lr:.6f} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f if val_loss else 'N/A'} - "
                f"Active Triplets: {train_stats['active_triplets_ratio']:.2%}"
            )
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, optimizer, train_loss)
                
            # Update history
            self.history['train_loss'].append(train_loss)
            if val_loss:
                self.history['val_loss'].append(val_loss)
            self.history['mining_stats'].append(train_stats)
            
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                     epoch: int) -> Tuple[float, Dict]:
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0
        total_samples = 0
        epoch_stats = defaultdict(float)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            loss, info = self.model.compute_loss(images, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update statistics
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Aggregate mining statistics
            for key, value in info.items():
                epoch_stats[key] += value
                
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'active': f"{info.get('active_triplets', 0)}/{info.get('total_triplets', 1)}"
            })
            
        # Compute epoch statistics
        avg_loss = total_loss / total_samples
        num_batches = len(train_loader)
        
        for key in epoch_stats:
            epoch_stats[key] /= num_batches
            
        # Add ratio of active triplets
        if epoch_stats['total_triplets'] > 0:
            epoch_stats['active_triplets_ratio'] = (
                epoch_stats['active_triplets'] / epoch_stats['total_triplets']
            )
        else:
            epoch_stats['active_triplets_ratio'] = 0
            
        return avg_loss, dict(epoch_stats)
    
    def validate(self, val_dataset: VGGFace2Dataset) -> float:
        """Validate the model using triplet-aware sampling."""
        self.model.eval()
        
        # Create validation batch sampler (smaller batches for validation)
        val_batch_sampler = FaceNetBatchSampler(
            val_dataset, 
            faces_per_identity=10,  # Smaller for validation
            num_identities_per_batch=20  # 200 total batch size
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                loss, _ = self.model.compute_loss(images, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Limit validation to a few batches for speed
                if num_batches >= 10:
                    break
                
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def adjust_learning_rate(self, optimizer: optim.Optimizer, epoch: int, 
                             lr_schedule: Dict[int, float]) -> float:
        """Adjust learning rate based on schedule.
        
        Returns:
            The current learning rate after adjustment
        """        
        current_lr = optimizer.param_groups[0]['lr']  # Get current LR as default
        
        # Find the appropriate learning rate for current epoch
        for epoch_threshold, new_lr in sorted(lr_schedule.items(), reverse=True):
            if epoch >= epoch_threshold:
                current_lr = new_lr
                break
                
        # Apply the learning rate to all parameter groups
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        return current_lr
            
    def save_checkpoint(self, epoch: int, optimizer: optim.Optimizer, loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'history': self.history
        }
        
        path = self.checkpoint_dir / f'facenet_epoch_{epoch+1}.pth'
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")
        
        # Save training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)