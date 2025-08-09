import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Union
from collections import defaultdict
from tqdm import tqdm
import logging
import json
from pathlib import Path
from dataset import VGGFace2Dataset, FaceNetBatchSampler
from model import FaceNetInceptionResNetV2
from model import ExponentialMovingAverage
import gc

class FaceNetTrainer:
    """
    Trainer class for FaceNet model following paper specifications.
    """
    
    def __init__(self, model: FaceNetInceptionResNetV2, device: torch.device, 
                 checkpoint_dir: str, 
                 ema_decay: float, 
                 lr_schedule: dict = None):
        """
        Initialize trainer with EMA optimization.
        
        Args:
            model: FaceNet model
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            ema_decay: EMA decay rate (default: 0.9999)
            lr_schedule: Learning rate schedule dict mapping epoch to lr
        """
        self.model = model
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Learning rate schedule
        self.lr_schedule = lr_schedule
        
        # Store EMA parameters for lazy initialization
        self.ema_decay = ema_decay
        self.ema = None  # Will be initialized when needed (Lazy init)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'mining_stats': []
        }
    
    def ensure_ema_initialized(self):
        """Initialize EMA if not already done. Called lazily to avoid memory issues."""
        if self.ema is None:
            self.ema = ExponentialMovingAverage(self.model, decay=self.ema_decay, device=self.device)
      
    def get_learning_rate(self, epoch: int) -> float:
        """Get learning rate for given epoch based on schedule."""
        # Find the latest epoch <= current epoch in the schedule
        valid_epochs = [e for e in self.lr_schedule.keys() if e <= epoch]
        if not valid_epochs:
            # If no valid epoch found, use the first entry
            return list(self.lr_schedule.values())[0]
        
        latest_epoch = max(valid_epochs)
        lr = self.lr_schedule[latest_epoch]
        
        # Check if this indicates end of training
        if lr == -1:
            return -1
        
        return lr
    
    def create_optimizer(self, learning_rate: float, weight_decay: float, 
                        optimizer_params: Dict) -> optim.Optimizer:
        """
        Create optimizer based on configuration.
        
        Args:
            learning_rate: Learning rate
            weight_decay: Weight decay
            optimizer_params: Optimizer parameters
            
        Returns:
            Configured optimizer
        """
        optimizer_type = optimizer_params.get('type', 'rmsprop')
        
        if optimizer_type.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                betas=optimizer_params.get('betas', [0.9, 0.999]),
                eps=optimizer_params.get('eps', 1e-8),
                weight_decay=weight_decay,
                amsgrad=optimizer_params.get('amsgrad', False)
            )
        elif optimizer_type.lower() == 'rmsprop':
            return optim.RMSprop(
                self.model.parameters(),
                lr=learning_rate,
                alpha=optimizer_params.get('alpha', 0.99),
                eps=optimizer_params.get('eps', 1e-8),
                weight_decay=weight_decay,
                momentum=optimizer_params.get('momentum', 0),
                centered=optimizer_params.get('centered', False)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Choose 'adam' or 'rmsprop'")
        
    def train(self, train_dataset: VGGFace2Dataset, 
              val_dataset: Optional[VGGFace2Dataset],
              num_epochs: int,
              learning_rate: float,
              faces_per_identity: int,
              num_identities_per_batch: int,
              weight_decay: float,
              optimizer_params: Dict = None):
        """
        Train the FaceNet model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate (InceptionResNetV2 paper uses 0.045)
            faces_per_identity: Faces per identity per batch
            num_identities_per_batch: Number of identities per batch
            weight_decay: Weight decay for regularization
            optimizer_params: Optimizer parameters
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
        
        # Create optimizer
        optimizer = self.create_optimizer(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_params=optimizer_params
        )
        
        # Training loop
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Optimizer: {optimizer_params.get('type', 'rmsprop').upper()}")
        self.logger.info(f"Batch size: {batch_sampler.batch_size} "
                        f"({faces_per_identity} faces x {num_identities_per_batch} identities)")
        
        for epoch in range(num_epochs):
            # Get learning rate for this epoch
            current_lr = self.get_learning_rate(epoch)
            
            # Check if we should stop training
            if current_lr == -1:
                self.logger.info(f"Learning rate schedule indicates end of training at epoch {epoch}")
                break
            
            # Update optimizer learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Train one epoch
            train_loss, train_stats = self.train_epoch(train_loader, optimizer, epoch)
            
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
                self.save_checkpoint(epoch, optimizer, train_loss)
                
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
            self.ensure_ema_initialized()
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
        self.ensure_ema_initialized()
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
            if self.ema is not None:
                self.ema.restore(self.model)
                
        return total_loss / num_batches if num_batches > 0 else 0.0
            
    def save_checkpoint(self, epoch: int, optimizer: optim.Optimizer, loss: float):
        """Save model checkpoint with EMA state."""
        self.ensure_ema_initialized()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_schedule': self.lr_schedule,
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
        Load model checkpoint with EMA state with memory optimization.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """     
        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load checkpoint with CPU mapping to reduce GPU memory pressure
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Move model to CPU temporarily to avoid device mismatch during loading
        self.model.cpu()
        
        # Extract model state dict and load
        model_state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(model_state_dict, strict=True)
        
        # Now move the loaded model back to GPU
        self.model.to(self.device)
        
        # Clear intermediate variables and collect garbage
        del model_state_dict
        gc.collect()
        
        # Clear GPU cache after model transfer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load EMA state if available
        if 'ema_state_dict' in checkpoint:
            # Initialize EMA from the loaded model (avoids duplicate creation)
            self.ema = ExponentialMovingAverage(self.model, decay=self.ema_decay, device=self.device)
            
            # Extract EMA state dict
            ema_state_dict = checkpoint['ema_state_dict']
            self.ema.load_state_dict(ema_state_dict)
            
            # Clear EMA state dict and collect garbage
            del ema_state_dict
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # Initialize EMA from current model if not in checkpoint
            self.ema = ExponentialMovingAverage(self.model, decay=self.ema_decay, device=self.device)
        
        # Load training history if available
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        # Load learning rate schedule if available
        if 'lr_schedule' in checkpoint:
            self.lr_schedule = checkpoint['lr_schedule']
        
        # Store essential checkpoint info
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', 0.0)
        }
        
        # Clear the full checkpoint from memory
        del checkpoint
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Checkpoint loading completed")
        return checkpoint_info