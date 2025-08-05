import torch
import argparse
import os
from pathlib import Path
import torchvision.transforms as transforms
from dataset import VGGFace2Dataset
from model import FaceNetInceptionResNetV2
from tools import FaceNetTrainer

def main(args):
    """Main training script."""
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Validate paths
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise ValueError(f"Data root directory does not exist: {data_root}")
    
    train_dir = data_root / 'train'
    val_dir = data_root / 'val'
    
    if not train_dir.exists():
        raise ValueError(f"Training directory does not exist: {train_dir}")
    if not val_dir.exists():
        print(f"Warning: Validation directory does not exist: {val_dir}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Data root: {data_root}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Embedding size: {args.embedding_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.faces_per_identity * args.num_identities_per_batch}")
    print(f"  EMA decay: {args.ema_decay}")
    print(f"  EMA enabled: True")
    
    # Create datasets with augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((299, 299)),  # InceptionResNetV2 input size
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = VGGFace2Dataset(args.data_root, split='train', transform=train_transforms)
    val_dataset = VGGFace2Dataset(args.data_root, split='val', transform=val_transforms) if val_dir.exists() else None
    
    # Create model
    model = FaceNetInceptionResNetV2(device=device, embedding_size=args.embedding_size)
    
    # Create trainer with EMA
    trainer = FaceNetTrainer(model, device, checkpoint_dir, ema_decay=args.ema_decay)
    
    # Resume from checkpoint if specified
    if args.resume:
        if not os.path.exists(args.resume):
            raise ValueError(f"Checkpoint file does not exist: {args.resume}")
        print(f"Resuming training from: {args.resume}")
        checkpoint = trainer.load_checkpoint(args.resume)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print("EMA state loaded from checkpoint")
    
    # Train model
    print("Starting training...")
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        faces_per_identity=args.faces_per_identity,
        num_identities_per_batch=args.num_identities_per_batch
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train FaceNet model')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to VGGFace2 dataset root directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--embedding_size', type=int, default=512,
                        help='Size of face embeddings (paper uses 512)')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='Number of training epochs (paper uses 1000+)')
    parser.add_argument('--learning_rate', type=float, default=0.045,
                        help='Initial learning rate (InceptionResNetV2 paper uses 0.045)')
    parser.add_argument('--faces_per_identity', type=int, default=40,
                        help='Number of faces per identity per batch (paper uses ~40)')
    parser.add_argument('--num_identities_per_batch', type=int, default=45,
                        help='Number of identities per batch (45*40=1800 total batch size)')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='EMA decay rate for model parameters (default: 0.9999)')
    
    args = parser.parse_args()
    main(args)