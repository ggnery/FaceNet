#!/usr/bin/env python3
"""
LFW Benchmark Script for FaceNet Model
Evaluates trained FaceNet model on Labeled Faces in the Wild (LFW) dataset
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import KFold
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import FaceNetInceptionResNetV2


class LFWBenchmark:
    """LFW benchmark evaluation for FaceNet model."""
    
    def __init__(self, model_path: str, lfw_data_path: str, device: str = 'auto'):
        """
        Initialize LFW benchmark.
        
        Args:
            model_path: Path to trained FaceNet model checkpoint
            lfw_data_path: Path to LFW dataset directory
            device: Device to run evaluation on ('auto', 'cuda', 'cpu')
        """
        self.device = self._get_device(device)
        self.lfw_data_path = Path(lfw_data_path)
        self.image_dir = self.lfw_data_path / 'lfw-deepfunneled' / 'lfw-deepfunneled'
        
        # Load model
        print(f"Loading FaceNet model from {model_path}")
        self.model = self._load_model(model_path)
        
        # Data transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print(f"Using device: {self.device}")
        print(f"LFW image directory: {self.image_dir}")
        
    def _get_device(self, device: str) -> torch.device:
        """Get computation device."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained FaceNet model."""
        model = FaceNetInceptionResNetV2(
            device=self.device,
            embedding_size=512
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # Training checkpoint with metadata
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            loss = checkpoint.get('loss', 'unknown')
            print(f"Loaded checkpoint from epoch {epoch} with loss {loss}")
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
            
        model.eval()
        
        return model
    
    def _load_image(self, person_name: str, image_num: int) -> torch.Tensor:
        """
        Load and preprocess LFW image.
        
        Args:
            person_name: Name of the person
            image_num: Image number for the person
            
        Returns:
            Preprocessed image tensor
        """
        # LFW image naming convention: PersonName_XXXX.jpg
        image_filename = f"{person_name}_{image_num:04d}.jpg"
        image_path = self.image_dir / person_name / image_filename
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)  # Add batch dimension
    
    def _get_embedding(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Get normalized embedding for an image.
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Normalized embedding vector
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            embedding = self.model(image_tensor)
            # L2 normalize the embedding
            embedding = F.normalize(embedding, p=2, dim=1)
            return embedding.cpu().numpy().flatten()
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def load_lfw_pairs(self) -> tuple:
        """
        Load LFW pairs from CSV file.
        
        Returns:
            Tuple of (pairs_data, labels) where:
            - pairs_data: List of tuples (person1, img1, person2, img2)
            - labels: List of labels (1 for same person, 0 for different)
        """
        pairs_file = self.lfw_data_path / 'pairs.csv'
        
        # Read the CSV file line by line to handle inconsistent formatting
        pairs_data = []
        labels = []
        
        print("Loading LFW pairs...")
        with open(pairs_file, 'r') as f:
            lines = f.readlines()
            
        # Skip header
        for line in tqdm(lines[1:], desc="Processing pairs"):
            line = line.strip()
            if not line:
                continue
            
            # Split by comma and remove empty strings
            parts = [part.strip() for part in line.split(',') if part.strip()]
            
            if len(parts) == 3:
                # Positive pair: same person, different images
                # Format: name, img1_num, img2_num
                person1 = parts[0]
                img1_num = int(parts[1])
                person2 = parts[0]  # Same person
                img2_num = int(parts[2])
                
                pairs_data.append((person1, img1_num, person2, img2_num))
                labels.append(1)  # Same person
                
            elif len(parts) == 4:
                # Negative pair: different people
                # Format: person1_name, img1_num, person2_name, img2_num
                person1 = parts[0]
                img1_num = int(parts[1])
                person2 = parts[2]
                img2_num = int(parts[3])
                
                pairs_data.append((person1, img1_num, person2, img2_num))
                labels.append(0)  # Different people
            else:
                print(f"Skipping malformed line: {line}")
                continue
        
        print(f"Loaded {len(pairs_data)} pairs ({sum(labels)} positive, {len(labels) - sum(labels)} negative)")
        return pairs_data, labels
    
    def evaluate_pairs(self, pairs_data: list, labels: list) -> tuple:
        """
        Evaluate model on LFW pairs.
        
        Args:
            pairs_data: List of image pairs
            labels: Ground truth labels
            
        Returns:
            Tuple of (similarities, labels, skipped_pairs)
        """
        similarities = []
        valid_labels = []
        skipped_pairs = 0
        
        print("Computing embeddings and similarities...")
        for i, (person1, img1, person2, img2) in enumerate(tqdm(pairs_data)):
            try:
                # Load images
                image1 = self._load_image(person1, img1)
                image2 = self._load_image(person2, img2)
                
                # Get embeddings
                emb1 = self._get_embedding(image1)
                emb2 = self._get_embedding(image2)
                
                # Compute similarity
                similarity = self._compute_similarity(emb1, emb2)
                similarities.append(similarity)
                valid_labels.append(labels[i])
                
            except (FileNotFoundError, Exception) as e:
                print(f"Skipping pair {i+1}: {e}")
                skipped_pairs += 1
                continue
        
        print(f"Processed {len(similarities)} pairs, skipped {skipped_pairs} pairs")
        return np.array(similarities), np.array(valid_labels), skipped_pairs
    
    def compute_metrics(self, similarities: np.ndarray, labels: np.ndarray) -> dict:
        """
        Compute evaluation metrics.
        
        Args:
            similarities: Array of similarity scores
            labels: Array of ground truth labels
            
        Returns:
            Dictionary of metrics
        """
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Compute accuracy at optimal threshold
        predictions = (similarities >= optimal_threshold).astype(int)
        accuracy = accuracy_score(labels, predictions)
        
        # Compute true positive rate and false positive rate at optimal threshold
        optimal_tpr = tpr[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        
        metrics = {
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'optimal_threshold': optimal_threshold,
            'optimal_tpr': optimal_tpr,
            'optimal_fpr': optimal_fpr,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
        
        return metrics
    
    def plot_results(self, metrics: dict, similarities: np.ndarray, labels: np.ndarray, save_path: str = None):
        """
        Plot evaluation results.
        
        Args:
            metrics: Computed metrics dictionary
            similarities: Array of similarity scores
            labels: Array of ground truth labels
            save_path: Path to save plots (optional)
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curve
        ax1.plot(metrics['fpr'], metrics['tpr'], 'b-', linewidth=2,
                label=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})')
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        ax1.plot(metrics['optimal_fpr'], metrics['optimal_tpr'], 'ro', markersize=8,
                label=f'Optimal Point (threshold = {metrics["optimal_threshold"]:.3f})')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Similarity Distribution
        pos_similarities = similarities[labels == 1]
        neg_similarities = similarities[labels == 0]
        
        ax2.hist(neg_similarities, bins=50, alpha=0.7, label='Different People', color='red', density=True)
        ax2.hist(pos_similarities, bins=50, alpha=0.7, label='Same Person', color='blue', density=True)
        ax2.axvline(metrics['optimal_threshold'], color='green', linestyle='--', linewidth=2,
                   label=f'Optimal Threshold ({metrics["optimal_threshold"]:.3f})')
        ax2.set_xlabel('Cosine Similarity')
        ax2.set_ylabel('Density')
        ax2.set_title('Similarity Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Accuracy vs Threshold
        accuracies = []
        threshold_range = np.linspace(similarities.min(), similarities.max(), 100)
        for thresh in threshold_range:
            preds = (similarities >= thresh).astype(int)
            acc = accuracy_score(labels, preds)
            accuracies.append(acc)
        
        ax3.plot(threshold_range, accuracies, 'g-', linewidth=2)
        ax3.axvline(metrics['optimal_threshold'], color='red', linestyle='--', linewidth=2,
                   label=f'Optimal Threshold')
        ax3.axhline(metrics['accuracy'], color='red', linestyle='--', linewidth=2,
                   label=f'Max Accuracy ({metrics["accuracy"]:.4f})')
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy vs Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Confusion Matrix-like visualization
        from sklearn.metrics import confusion_matrix
        predictions = (similarities >= metrics['optimal_threshold']).astype(int)
        cm = confusion_matrix(labels, predictions)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                   xticklabels=['Different', 'Same'], yticklabels=['Different', 'Same'])
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        ax4.set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results saved to {save_path}")
        
        plt.show()
    
    def run_benchmark(self, save_results: bool = True) -> dict:
        """
        Run complete LFW benchmark evaluation.
        
        Args:
            save_results: Whether to save results to files
            
        Returns:
            Dictionary of evaluation results
        """
        print("=" * 60)
        print("FaceNet LFW Benchmark Evaluation")
        print("=" * 60)
        
        # Load LFW pairs
        pairs_data, labels = self.load_lfw_pairs()
        
        # Evaluate pairs
        similarities, valid_labels, skipped_pairs = self.evaluate_pairs(pairs_data, labels)
        
        if len(similarities) == 0:
            raise ValueError("No valid pairs found for evaluation!")
        
        # Compute metrics
        metrics = self.compute_metrics(similarities, valid_labels)
        
        # Print results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total pairs processed: {len(similarities)}")
        print(f"Pairs skipped: {skipped_pairs}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"Best Accuracy: {metrics['accuracy']:.4f}")
        print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
        print(f"True Positive Rate: {metrics['optimal_tpr']:.4f}")
        print(f"False Positive Rate: {metrics['optimal_fpr']:.4f}")
        
        # Plot results
        if save_results:
            save_path = Path('benchmark') / 'lfw_evaluation_results.png'
            save_path.parent.mkdir(exist_ok=True)
        else:
            save_path = None
            
        self.plot_results(metrics, similarities, valid_labels, save_path)
        
        # Save detailed results
        if save_results:
            results_file = Path('benchmark') / 'lfw_benchmark_results.npz'
            np.savez(results_file,
                    similarities=similarities,
                    labels=valid_labels,
                    **metrics)
            print(f"Detailed results saved to {results_file}")
        
        return {
            'similarities': similarities,
            'labels': valid_labels,
            'metrics': metrics,
            'skipped_pairs': skipped_pairs
        }


def main():
    """Main function for running LFW benchmark."""
    parser = argparse.ArgumentParser(description='LFW Benchmark for FaceNet')
    parser.add_argument('--model_path', type=str, 
                       default='checkpoints/facenet_epoch_500.pth',
                       help='Path to trained FaceNet model checkpoint')
    parser.add_argument('--lfw_data_path', type=str,
                       default='data/lfw',
                       help='Path to LFW dataset directory')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to run evaluation on')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results to files')
    
    args = parser.parse_args()
    
    # Validate paths
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    lfw_path = Path(args.lfw_data_path)
    if not lfw_path.exists():
        raise FileNotFoundError(f"LFW dataset not found: {lfw_path}")
    
    # Run benchmark
    benchmark = LFWBenchmark(
        model_path=str(model_path),
        lfw_data_path=str(lfw_path),
        device=args.device
    )
    
    results = benchmark.run_benchmark(save_results=not args.no_save)
    
    print("\nBenchmark completed successfully!")
    return results


if __name__ == '__main__':
    main() 