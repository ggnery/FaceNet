import torch
import os
from pathlib import Path
import torchvision.transforms as transforms
from dataset import VGGFace2Dataset
from model import FaceNetInceptionResNetV2
from tools import FaceNetTrainer
from tools.config import Config

def main(config: Config):
    """Main training script."""
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Validate paths
    data_root = Path(config.data_root)
    if not data_root.exists():
        raise ValueError(f"Data root directory does not exist: {data_root}")
    
    train_dir = data_root / config.train_dir
    val_dir = data_root / config.val_dir
    
    if not train_dir.exists():
        raise ValueError(f"Training directory does not exist: {train_dir}")
    if not val_dir.exists():
        print(f"Warning: Validation directory does not exist: {val_dir}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Display configuration
    config.print_config()
    
    # Create datasets with augmentation
    train_transforms = transforms.Compose([
        transforms.Resize(tuple(config.input_size)),  # InceptionResNetV2 input size
        transforms.RandomHorizontalFlip(p=config.random_horizontal_flip),
        transforms.RandomRotation(degrees=config.random_rotation),
        transforms.ColorJitter(
            brightness=config.color_jitter_brightness, 
            contrast=config.color_jitter_contrast, 
            saturation=config.color_jitter_saturation
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.norm_mean, std=config.norm_std)
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(tuple(config.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.norm_mean, std=config.norm_std)
    ])
     
    train_dataset = VGGFace2Dataset(config.data_root, split='train', transform=train_transforms)
    val_dataset = VGGFace2Dataset(config.data_root, split='val', transform=val_transforms) if val_dir.exists() else None
    
    # Create model
    model = FaceNetInceptionResNetV2(device=device, embedding_size=config.embedding_size, margin=config.margin, dropout_keep=config.dropout_keep)
    
    # Create trainer with EMA and learning rate schedule
    trainer = FaceNetTrainer(
        model, 
        device, 
        checkpoint_dir, 
        ema_decay=config.ema_decay,
        lr_schedule=config.lr_schedule
    )
    
    # Resume from checkpoint
    if config.resume_checkpoint:
        if not os.path.exists(config.resume_checkpoint):
            raise ValueError(f"Checkpoint file does not exist: {config.resume_checkpoint}")
        print(f"Resuming training from: {config.resume_checkpoint}")
        checkpoint = trainer.load_checkpoint(config.resume_checkpoint)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print("EMA state loaded from checkpoint")
    
    # Train model
    print("Starting training...")
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    
    # Optimizer parameters
    optimizer_params = {}
    
    if config.optimizer_type == 'adam':
        optimizer_params.update({
            'type': config.optimizer_type,
            'betas': config.adam_betas,
            'eps': config.adam_eps,
            'amsgrad': config.adam_amsgrad
        })
    elif config.optimizer_type == 'rmsprop':
        optimizer_params.update({
            'type': config.optimizer_type,
            'alpha': config.rmsprop_alpha,
            'eps': config.rmsprop_eps,
            'momentum': config.rmsprop_momentum,
            'centered': config.rmsprop_centered
        })
    
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        faces_per_identity=config.faces_per_identity,
        num_identities_per_batch=config.num_identities_per_batch,
        weight_decay=config.weight_decay,
        optimizer_params=optimizer_params
    )


if __name__ == "__main__":
    # Load configuration from YAML file
    config = Config("config.yml")
    main(config)