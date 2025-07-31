#!/usr/bin/env python3
"""
Example usage script for FaceNet training.
Shows how to run training with different configurations.
"""

import subprocess
import sys

def run_training_example():
    """Example of how to run the training script."""
    
    # Example 1: Basic training
    print("Example 1: Basic training with default parameters")
    cmd = [
        sys.executable, "train.py",
        "--data_root", "/path/to/your/vggface2",  # Replace with your actual path
        "--checkpoint_dir", "./checkpoints",
        "--embedding_size", "512",
        "--num_epochs", "1000",
        "--learning_rate", "0.05"
    ]
    print("Command:", " ".join(cmd))
    print()
    
    # Example 2: Resume training
    print("Example 2: Resume training from checkpoint")
    cmd_resume = [
        sys.executable, "train.py", 
        "--data_root", "/path/to/your/vggface2",
        "--resume", "./checkpoints/facenet_epoch_100.pth",
        "--num_epochs", "2000"
    ]
    print("Command:", " ".join(cmd_resume))
    print()
    
    # Example 3: Smaller batch size for limited GPU memory
    print("Example 3: Smaller batch size for limited GPU memory")
    cmd_small = [
        sys.executable, "train.py",
        "--data_root", "/path/to/your/vggface2",
        "--faces_per_identity", "20",  # Reduced from 40
        "--num_identities_per_batch", "25",  # 20 * 25 = 500 batch size
        "--embedding_size", "512"
    ]
    print("Command:", " ".join(cmd_small))
    print()
    
    # Example 4: Quick test run
    print("Example 4: Quick test run with fewer epochs")
    cmd_test = [
        sys.executable, "train.py",
        "--data_root", "/path/to/your/vggface2", 
        "--num_epochs", "10",
        "--faces_per_identity", "10",
        "--num_identities_per_batch", "10"  # 10 * 10 = 100 batch size
    ]
    print("Command:", " ".join(cmd_test))
    print()

def print_usage_instructions():
    """Print detailed usage instructions."""
    print("="*60)
    print("FaceNet Training Usage Instructions")
    print("="*60)
    print()
    print("1. First, prepare your VGGFace2 dataset:")
    print("   - Download VGGFace2 dataset")
    print("   - Organize it as:")
    print("     /path/to/vggface2/")
    print("     ├── train/")
    print("     │   ├── identity1/")
    print("     │   │   ├── img1.jpg")
    print("     │   │   └── img2.jpg")
    print("     │   └── identity2/")
    print("     └── test/")
    print("         ├── identity1/")
    print("         └── identity2/")
    print()
    print("2. Install requirements:")
    print("   pip install torch torchvision tqdm Pillow")
    print()
    print("3. Run training:")
    print("   python train.py --data_root /path/to/your/vggface2")
    print()
    print("4. Monitor training:")
    print("   - Checkpoints saved to ./checkpoints/")
    print("   - Training history saved as JSON")
    print("   - Resume with: --resume ./checkpoints/facenet_epoch_N.pth")
    print()
    print("5. Adjust for your hardware:")
    print("   - Large GPU (24GB+): Default settings (1800 batch size)")
    print("   - Medium GPU (8-12GB): --faces_per_identity 20 --num_identities_per_batch 25")
    print("   - Small GPU (4-6GB): --faces_per_identity 10 --num_identities_per_batch 20")
    print()

if __name__ == "__main__":
    print_usage_instructions()
    run_training_example()
    
    print("="*60)
    print("To actually run training, replace '/path/to/your/vggface2' with your actual dataset path!")
    print("="*60) 