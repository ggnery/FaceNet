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
from model import ExponentialMovingAverage

class FaceNetTrainer:
    """
    Trainer class for FaceNet model following paper specifications.
    """
    
    def __init__(self, model: FaceNetInceptionResNetV2, device: torch.device, 
                 checkpoint_dir: str, 
                 ema_decay: float):
        """
        Initialize trainer with EMA optimization.
        
        Args:
            model: FaceNet model
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            ema_decay: EMA decay rate (default: 0.9999)
        """
        self.model = model
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize EMA
        self.ema = ExponentialMovingAverage(model, decay=ema_decay, device=device)
        
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
              val_dataset: Optional[VGGFace2Dataset],
              num_epochs: int,
              learning_rate: float,
              faces_per_identity: int,
              num_identities_per_batch: int,
              weight_decay: float):
        """
        Train the FaceNet model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate (InceptionResNetV2 paper uses 0.045)
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
        
        # Optimizer
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler 
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[500, 1000],  
            gamma=0.5  
        )
        
        # Training loop
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Batch size: {batch_sampler.batch_size} "
                        f"({faces_per_identity} faces x {num_identities_per_batch} identities)")
        
        for epoch in range(num_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            
            # Train one epoch
            train_loss, train_stats = self.train_epoch(train_loader, optimizer, epoch)
            
            scheduler.step()
            
            # Validation
            if val_dataset:
                val_loss = self.validate(val_dataset, faces_per_identity, num_identities_per_batch)
                
            # Log progress
            val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"LR: {current_lr:.6f} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss_str} - "
                f"Active Triplets: {train_stats['active_triplets_ratio']:.2%}"
            )
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, optimizer, scheduler, train_loss)
                
            # Update history
            self.history['train_loss'].append(train_loss)
            if val_loss:
                self.history['val_loss'].append(val_loss)
            self.history['mining_stats'].append(train_stats)
            
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, epoch: int) -> Tuple[float, Dict]:
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0
        total_samples = 0
        epoch_stats = defaultdict(float)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for _, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            loss, info = self.model.compute_loss(images, labels)
            
            # Skip batch if no valid triplets found
            if loss.item() == 0.0 and info.get('total_triplets', 0) == 0:
                continue
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update EMA after optimizer step
            self.ema.update(self.model)
            
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
            
        # epoch statistics
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
    
    def validate(self, val_dataset: VGGFace2Dataset, 
                 faces_per_identity: int,
                 num_identities_per_batch: int) -> float:
        """Validate the model using EMA parameters and triplet-aware sampling."""
        self.model.eval()
        
        # Apply EMA parameters for validation
        self.ema.apply_shadow(self.model)
        
        try:
            # Create validation batch sampler (smaller batches for validation)
            val_batch_sampler = FaceNetBatchSampler(
                val_dataset, 
                faces_per_identity=faces_per_identity,
                num_identities_per_batch=num_identities_per_batch
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_sampler=val_batch_sampler,
                num_workers=8,
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
                        
        finally:
            # Always restore original parameters
            self.ema.restore(self.model)
                
        return total_loss / num_batches if num_batches > 0 else 0.0
            
    def save_checkpoint(self, epoch: int, optimizer: optim.Optimizer, 
                       scheduler: optim.lr_scheduler.StepLR, loss: float):
        """Save model checkpoint with EMA state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
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
            
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint with EMA state.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load EMA state if available
        if 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
        else:
            self.logger.warning("No EMA state found in checkpoint, initializing EMA from current model")
            # Re-initialize EMA from current model if not in checkpoint
            self.ema = ExponentialMovingAverage(self.model, decay=self.ema.decay, device=self.device)
        
        # Load training history if available
        if 'history' in checkpoint:
            self.history = checkpoint['history']
            
        return checkpoint