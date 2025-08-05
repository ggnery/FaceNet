
import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import VGGFace2Dataset, FaceNetBatchSampler
import torch
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time
from PIL import Image

def test_dataset_initialization(dataset_path, split='train', mtcnn_weights=None):
    """Test basic dataset initialization and structure."""
    print("=" * 50)
    print("Testing Dataset Initialization")
    print("=" * 50)
    print(f"Using dataset path: {dataset_path}")
    print(f"Testing split: {split}")
    if mtcnn_weights:
        print(f"Using MTCNN weights: {mtcnn_weights}")
    
    # Test if dataset path exists
    if not Path(dataset_path).exists():
        print(f"[ERROR] Dataset path does not exist: {dataset_path}")
        return False
        
    # Test dataset initialization
    try:
        dataset = VGGFace2Dataset(dataset_path, split=split, mtcnn_weights=mtcnn_weights)
        print(f"[PASS] {split.capitalize()} dataset initialized successfully")
        print(f"   - Total samples: {len(dataset)}")
        print(f"   - Number of identities: {dataset.num_identities}")
        
        
        avg_samples_per_identity = len(dataset) / dataset.num_identities
        print(f"   - Average samples per identity: {avg_samples_per_identity:.2f}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize dataset: {e}")
        return False

def test_dataset_loading(dataset_path, split='train', mtcnn_weights=None):
    """Test individual sample loading and preprocessing."""
    print("\n" + "=" * 50)
    print("Testing Dataset Loading")
    print("=" * 50)
    
    try:
        dataset = VGGFace2Dataset(dataset_path, split=split, mtcnn_weights=mtcnn_weights)
        
        # Test loading first few samples
        for i in range(min(5, len(dataset))):
            try:
                image, label = dataset[i]
                print(f"[PASS] Sample {i}: shape={image.shape}, label={label}, dtype={image.dtype}")
                
                # Validate image properties
                if image.shape != (3, 299, 299):
                    print(f"[WARN] Expected shape (3, 299, 299), got {image.shape}")
                
                if image.dtype != torch.float32:
                    print(f"[WARN] Expected dtype float32, got {image.dtype}")
                    
                # Check value range (should be normalized to [-1, 1])
                min_val, max_val = image.min().item(), image.max().item()
                if min_val < -1.1 or max_val > 1.1:
                    print(f"[WARN] Values outside expected range [-1, 1]: [{min_val:.3f}, {max_val:.3f}]")
                    
            except Exception as e:
                print(f"[ERROR] Failed to load sample {i}: {e}")
                return False
                
        return True
    except Exception as e:
        print(f"[ERROR] Failed in dataset loading test: {e}")
        return False

def test_face_detection(dataset_path, split='train', mtcnn_weights=None):
    """Test face detection functionality."""
    print("\n" + "=" * 50)
    print("Testing Face Detection")
    print("=" * 50)
    
    try:      
        dataset = VGGFace2Dataset(dataset_path, split=split, mtcnn_weights=mtcnn_weights)
        
        # Test a few samples to see face detection results
        print("Testing face detection on sample images...")
        detection_stats = {'detected': 0, 'not_detected': 0}
        
        for i in range(min(10, len(dataset))):
            img_path, _ = dataset.samples[i]
            
            # Load original image
            original_image = Image.open(img_path).convert('RGB')
            boxes, probs = dataset.mtcnn.detect(original_image)
            
            if boxes is not None and len(boxes) > 0:
                detection_stats['detected'] += 1
                confidence = max(probs)
                print(f"[PASS] Sample {i}: Face detected (confidence: {confidence:.3f})")
            else:
                detection_stats['not_detected'] += 1
                print(f"[WARN] Sample {i}: No face detected")
        
        detection_rate = detection_stats['detected'] / (detection_stats['detected'] + detection_stats['not_detected'])
        print(f"\nFace detection rate: {detection_rate:.2%}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed in face detection test: {e}")
        return False

def test_batch_sampler(dataset_path, split='train', mtcnn_weights=None):
    """Test the FaceNet batch sampler."""
    print("\n" + "=" * 50)
    print("Testing Batch Sampler")
    print("=" * 50)
    
    try:
        dataset = VGGFace2Dataset(dataset_path, split=split, mtcnn_weights=mtcnn_weights)
        
        # Create batch sampler
        sampler = FaceNetBatchSampler(dataset, faces_per_identity=5, num_identities_per_batch=10)
        
        print(f"[PASS] Batch sampler initialized")
        print(f"   - Expected batch size: {sampler.batch_size}")
        print(f"   - Valid identities: {len(sampler.valid_identities)}")
        print(f"   - Number of batches per epoch: {len(sampler)}")
        
        # Test a few batches
        batch_count = 0
        for batch_indices in sampler:
            batch_count += 1
            
            # Get labels for this batch
            batch_labels = [dataset.samples[idx][1] for idx in batch_indices]
            label_counts = Counter(batch_labels)
            
            print(f"\nBatch {batch_count}:")
            print(f"   - Batch size: {len(batch_indices)}")
            print(f"   - Unique identities: {len(label_counts)}")
            print(f"   - Samples per identity: {dict(list(label_counts.items())[:5])}...")
            
            # Validate batch structure
            if len(batch_indices) != sampler.batch_size:
                print(f"[WARN] Expected batch size {sampler.batch_size}, got {len(batch_indices)}")
                
        return True
    except Exception as e:
        print(f"[ERROR] Failed in batch sampler test: {e}")
        return False

def test_data_distribution(dataset_path, split='train', mtcnn_weights=None):
    """Analyze the distribution of data across identities."""
    print("\n" + "=" * 50)
    print("Testing Data Distribution")
    print("=" * 50)
    
    try:
        dataset = VGGFace2Dataset(dataset_path, split=split, mtcnn_weights=mtcnn_weights)
        
        # Analyze samples per identity
        samples_per_identity = [len(indices) for indices in dataset.label_to_indices.values()]
        
        print(f"Data distribution statistics:")
        print(f"   - Min samples per identity: {min(samples_per_identity)}")
        print(f"   - Max samples per identity: {max(samples_per_identity)}")
        print(f"   - Mean samples per identity: {np.mean(samples_per_identity):.2f}")
        print(f"   - Median samples per identity: {np.median(samples_per_identity):.2f}")
        print(f"   - Std samples per identity: {np.std(samples_per_identity):.2f}")
        
        # Count identities with different sample ranges
        ranges = {
            '1-5': sum(1 for x in samples_per_identity if 1 <= x <= 5),
            '6-10': sum(1 for x in samples_per_identity if 6 <= x <= 10),
            '11-20': sum(1 for x in samples_per_identity if 11 <= x <= 20),
            '21-50': sum(1 for x in samples_per_identity if 21 <= x <= 50),
            '50+': sum(1 for x in samples_per_identity if x > 50)
        }
        
        print(f"\nIdentity distribution by sample count:")
        for range_name, count in ranges.items():
            percentage = count / len(samples_per_identity) * 100
            print(f"   - {range_name} samples: {count} identities ({percentage:.1f}%)")
            
        return True
    except Exception as e:
        print(f"[ERROR] Failed in data distribution test: {e}")
        return False

def test_performance(dataset_path, split='train', mtcnn_weights=None):
    """Test dataset loading performance."""
    print("\n" + "=" * 50)
    print("Testing Performance")
    print("=" * 50)
    
    try:
        dataset = VGGFace2Dataset(dataset_path, split=split, mtcnn_weights=mtcnn_weights)
        
        # Test loading speed
        num_samples = min(50, len(dataset))
        print(f"Testing loading speed for {num_samples} samples...")
        
        start_time = time.time()
        for i in range(num_samples):
            _ = dataset[i]
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_samples
        print(f"[PASS] Average loading time per sample: {avg_time:.4f} seconds")
        print(f"   - Estimated time for 1000 samples: {avg_time * 1000:.2f} seconds")
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed in performance test: {e}")
        return False

def show_image_samples(dataset_path, split='train', num_samples=8, mtcnn_weights=None):
    """Display sample images from the dataset with metadata."""
    print("\n" + "=" * 50)
    print("Showing Image Samples")
    print("=" * 50)
    
    try:
        dataset = VGGFace2Dataset(dataset_path, split=split, mtcnn_weights=mtcnn_weights)
        
        # Select random samples
        sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        # Create figure for displaying images
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        print(f"Displaying {len(sample_indices)} random samples...")
        
        for i, idx in enumerate(sample_indices):
            if i >= len(axes):
                break
                
            # Get sample data
            image, label = dataset[idx]
            img_path, _ = dataset.samples[idx]
            
            # Convert tensor to displayable format
            # Images are normalized to [-1, 1], convert to [0, 1] for display
            display_image = (image.permute(1, 2, 0) + 1) / 2
            display_image = torch.clamp(display_image, 0, 1)
            
            # Load original image for comparison
            original_image = Image.open(img_path).convert('RGB')
            
            # Display processed image
            axes[i].imshow(display_image)
            axes[i].set_title(f'Sample {idx}\nLabel: {label}\nShape: {image.shape}', fontsize=9)
            axes[i].axis('off')
            
            # Print detailed info
            print(f"Sample {idx}:")
            print(f"   - Path: {Path(img_path).name}")
            print(f"   - Label: {label}")
            print(f"   - Original size: {original_image.size}")
            print(f"   - Processed shape: {image.shape}")
            print(f"   - Value range: [{image.min().item():.3f}, {image.max().item():.3f}]")
            
            # Test face detection on original image
            try:
                boxes, probs = dataset.mtcnn.detect(original_image)
                if boxes is not None and len(boxes) > 0:
                    best_confidence = max(probs)
                    print(f"   - Face detected: confidence={best_confidence:.3f}")
                else:
                    print(f"   - No face detected")
            except Exception as e:
                print(f"   - Face detection error: {e}")
            print()
        
        # Hide unused subplots
        for i in range(len(sample_indices), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
        print(f"[PASS] Sample images saved to 'dataset_samples.png'")
        plt.show()
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to show image samples: {e}")
        return False

def run_all_tests(dataset_path, split='train', mtcnn_weights=None):
    """Run all dataset tests."""
    print("Starting Dataset Debug Tests")
    print("=" * 60)
    
    tests = [
        test_dataset_initialization,
        test_dataset_loading,
        test_face_detection,
        test_batch_sampler,
        test_data_distribution,
        test_performance,
        show_image_samples
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func(dataset_path=dataset_path, split=split, mtcnn_weights=mtcnn_weights)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "[PASS]" if result else "[FAIL]"
        print(f"{test_func.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("All tests passed! Dataset is ready for training.")
    else:
        print("Some tests failed. Please check the issues above.")

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Debug and test VGG Face2 dataset')
    parser.add_argument('dataset_path', 
                       help='Path to the VGG Face2 dataset directory')
    parser.add_argument('--split', 
                       default='train',
                       choices=['train', 'test'],
                       help='Dataset split to test (default: train)')
    parser.add_argument('--mtcnn-weights', 
                       default=None,
                       help='Path to MTCNN pretrained weights folder')
    parser.add_argument('--show-samples', 
                       action='store_true',
                       help='Only show image samples without running other tests')
    parser.add_argument('--num-samples', 
                       type=int,
                       default=8,
                       help='Number of sample images to display (default: 8)')
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not Path(args.dataset_path).exists():
        print(f"[ERROR] Dataset path does not exist: {args.dataset_path}")
        sys.exit(1)
    
    print(f"Testing dataset at: {args.dataset_path}")
    print(f"Split: {args.split}")
    if args.mtcnn_weights:
        print(f"MTCNN weights: {args.mtcnn_weights}")
    
    if args.show_samples:
        # Only show image samples
        show_image_samples(args.dataset_path, args.split, args.num_samples, args.mtcnn_weights)
    else:
        # Run all tests
        run_all_tests(args.dataset_path, args.split, args.mtcnn_weights)

if __name__ == "__main__":
    main()

