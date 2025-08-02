#!/usr/bin/env python3
"""
FaceNet Training Data Visualization Script

This script generates comprehensive graphs for FaceNet training data including:
- Training and validation loss curves
- Triplet mining statistics
- Distance metrics evolution
- Active triplets analysis
- Mining distribution over time
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Optional

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrainingVisualizer:
    """Visualizes FaceNet training data from training history JSON file."""
    
    def __init__(self, history_file: str):
        """
        Initialize visualizer with training history file.
        
        Args:
            history_file: Path to the training history JSON file
        """
        self.history_file = Path(history_file)
        self.data = self.load_data()
        self.setup_style()
        
    def load_data(self) -> Dict:
        """Load training data from JSON file."""
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
            print(f"Loaded training data with {len(data.get('train_loss', []))} epochs")
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Training history file not found: {self.history_file}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {self.history_file}")
    
    def setup_style(self):
        """Setup matplotlib style for consistent plots."""
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16
        })
    
    def plot_loss_curves(self, save_path: Optional[str] = None):
        """Plot training and validation loss curves."""
        train_loss = self.data.get('train_loss', [])
        val_loss = self.data.get('val_loss', [])
        
        if not train_loss and not val_loss:
            print("No loss data found in training history")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        epochs = range(1, len(train_loss) + 1)
        
        # Plot 1: Both losses on same plot
        if train_loss:
            ax1.plot(epochs, train_loss, label='Training Loss', linewidth=2, alpha=0.8)
        if val_loss:
            ax1.plot(epochs[:len(val_loss)], val_loss, label='Validation Loss', linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Loss difference (if both available)
        if train_loss and val_loss:
            min_len = min(len(train_loss), len(val_loss))
            loss_diff = np.array(val_loss[:min_len]) - np.array(train_loss[:min_len])
            ax2.plot(range(1, min_len + 1), loss_diff, 'r-', linewidth=2, alpha=0.8)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Validation - Training Loss')
            ax2.set_title('Overfitting Indicator')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Validation loss not available', 
                    transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title('Loss Difference (N/A)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_mining_statistics(self, save_path: Optional[str] = None):
        """Plot triplet mining statistics over time."""
        mining_stats = self.data.get('mining_stats', [])
        
        if not mining_stats:
            print("No mining statistics found in training history")
            return
        
        epochs = range(1, len(mining_stats) + 1)
        
        # Extract statistics
        semi_hard = [stat.get('semi_hard_negatives', 0) for stat in mining_stats]
        hard_neg = [stat.get('hard_negatives', 0) for stat in mining_stats]
        easy_neg = [stat.get('easy_negatives', 0) for stat in mining_stats]
        total_pairs = [stat.get('total_pairs', 0) for stat in mining_stats]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Absolute numbers
        ax1.plot(epochs, semi_hard, label='Semi-hard Negatives', linewidth=2)
        ax1.plot(epochs, hard_neg, label='Hard Negatives', linewidth=2)
        ax1.plot(epochs, easy_neg, label='Easy Negatives', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Count')
        ax1.set_title('Negative Types Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Percentages
        if total_pairs and total_pairs[0] > 0:
            semi_hard_pct = [s/t*100 for s, t in zip(semi_hard, total_pairs)]
            hard_neg_pct = [h/t*100 for h, t in zip(hard_neg, total_pairs)]
            easy_neg_pct = [e/t*100 for e, t in zip(easy_neg, total_pairs)]
            
            ax2.plot(epochs, semi_hard_pct, label='Semi-hard %', linewidth=2)
            ax2.plot(epochs, hard_neg_pct, label='Hard %', linewidth=2)
            ax2.plot(epochs, easy_neg_pct, label='Easy %', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Percentage')
            ax2.set_title('Negative Types Distribution (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Active triplets ratio
        active_ratio = [stat.get('active_triplets_ratio', 0) for stat in mining_stats]
        ax3.plot(epochs, active_ratio, 'g-', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Ratio')
        ax3.set_title('Active Triplets Ratio')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Plot 4: Valid pairs
        valid_pairs = [stat.get('valid_pairs', 0) for stat in mining_stats]
        ax4.plot(epochs, valid_pairs, 'purple', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Count')
        ax4.set_title('Valid Pairs')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_distance_metrics(self, save_path: Optional[str] = None):
        """Plot positive and negative distance evolution."""
        mining_stats = self.data.get('mining_stats', [])
        
        if not mining_stats:
            print("No mining statistics found for distance metrics")
            return
        
        epochs = range(1, len(mining_stats) + 1)
        
        avg_pos_dist = [stat.get('avg_pos_distance', 0) for stat in mining_stats]
        avg_neg_dist = [stat.get('avg_neg_distance', 0) for stat in mining_stats]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Distance evolution
        ax1.plot(epochs, avg_pos_dist, label='Avg Positive Distance', linewidth=2, color='blue')
        ax1.plot(epochs, avg_neg_dist, label='Avg Negative Distance', linewidth=2, color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Distance')
        ax1.set_title('Average Distances Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distance margin (negative - positive)
        margin = [neg - pos for pos, neg in zip(avg_pos_dist, avg_neg_dist)]
        ax2.plot(epochs, margin, 'green', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Distance Margin')
        ax2.set_title('Distance Margin (Negative - Positive)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_summary(self, save_path: Optional[str] = None):
        """Create a comprehensive summary plot."""
        fig = plt.figure(figsize=(20, 12))
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        train_loss = self.data.get('train_loss', [])
        val_loss = self.data.get('val_loss', [])
        mining_stats = self.data.get('mining_stats', [])
        
        epochs_loss = range(1, len(train_loss) + 1)
        epochs_mining = range(1, len(mining_stats) + 1)
        
        # 1. Loss curves
        ax1 = fig.add_subplot(gs[0, :2])
        if train_loss:
            ax1.plot(epochs_loss, train_loss, label='Training Loss', linewidth=2)
        if val_loss:
            ax1.plot(epochs_loss[:len(val_loss)], val_loss, label='Validation Loss', linewidth=2)
        ax1.set_title('Loss Evolution')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Active triplets ratio
        ax2 = fig.add_subplot(gs[0, 2])
        if mining_stats:
            active_ratio = [stat.get('active_triplets_ratio', 0) for stat in mining_stats]
            ax2.plot(epochs_mining, active_ratio, 'green', linewidth=2)
        ax2.set_title('Active Triplets Ratio')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Ratio')
        ax2.grid(True, alpha=0.3)
        
        # 3. Distance metrics
        ax3 = fig.add_subplot(gs[1, :])
        if mining_stats:
            avg_pos_dist = [stat.get('avg_pos_distance', 0) for stat in mining_stats]
            avg_neg_dist = [stat.get('avg_neg_distance', 0) for stat in mining_stats]
            margin = [neg - pos for pos, neg in zip(avg_pos_dist, avg_neg_dist)]
            
            ax3_twin = ax3.twinx()
            line1 = ax3.plot(epochs_mining, avg_pos_dist, 'blue', linewidth=2, label='Positive Distance')
            line2 = ax3.plot(epochs_mining, avg_neg_dist, 'red', linewidth=2, label='Negative Distance')
            line3 = ax3_twin.plot(epochs_mining, margin, 'green', linewidth=2, label='Margin')
            
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Distance')
            ax3_twin.set_ylabel('Margin')
            ax3.set_title('Distance Metrics Evolution')
            
            # Combine legends
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc='upper right')
            ax3.grid(True, alpha=0.3)
        
        # 4. Mining distribution
        ax4 = fig.add_subplot(gs[2, :])
        if mining_stats:
            semi_hard = [stat.get('semi_hard_negatives', 0) for stat in mining_stats]
            hard_neg = [stat.get('hard_negatives', 0) for stat in mining_stats]
            easy_neg = [stat.get('easy_negatives', 0) for stat in mining_stats]
            
            ax4.stackplot(epochs_mining, semi_hard, hard_neg, easy_neg,
                         labels=['Semi-hard', 'Hard', 'Easy'], alpha=0.8)
            ax4.set_title('Negative Mining Distribution (Stacked)')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Count')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('FaceNet Training Summary', fontsize=20, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_training_summary(self):
        """Print a text summary of training statistics."""
        train_loss = self.data.get('train_loss', [])
        val_loss = self.data.get('val_loss', [])
        mining_stats = self.data.get('mining_stats', [])
        
        print("\n" + "="*60)
        print("FACENET TRAINING SUMMARY")
        print("="*60)
        
        if train_loss:
            print(f"Total Epochs: {len(train_loss)}")
            print(f"Initial Training Loss: {train_loss[0]:.6f}")
            print(f"Final Training Loss: {train_loss[-1]:.6f}")
            print(f"Training Loss Improvement: {train_loss[0] - train_loss[-1]:.6f}")
            print(f"Best Training Loss: {min(train_loss):.6f} (Epoch {train_loss.index(min(train_loss)) + 1})")
        
        if val_loss:
            print(f"Initial Validation Loss: {val_loss[0]:.6f}")
            print(f"Final Validation Loss: {val_loss[-1]:.6f}")
            print(f"Validation Loss Improvement: {val_loss[0] - val_loss[-1]:.6f}")
            print(f"Best Validation Loss: {min(val_loss):.6f} (Epoch {val_loss.index(min(val_loss)) + 1})")
        
        if mining_stats:
            final_stats = mining_stats[-1]
            print(f"\nFinal Mining Statistics:")
            print(f"  Active Triplets Ratio: {final_stats.get('active_triplets_ratio', 0):.4f}")
            print(f"  Average Positive Distance: {final_stats.get('avg_pos_distance', 0):.4f}")
            print(f"  Average Negative Distance: {final_stats.get('avg_neg_distance', 0):.4f}")
            print(f"  Distance Margin: {final_stats.get('avg_neg_distance', 0) - final_stats.get('avg_pos_distance', 0):.4f}")
            print(f"  Semi-hard Negatives: {final_stats.get('semi_hard_negatives', 0):.1f}")
            print(f"  Hard Negatives: {final_stats.get('hard_negatives', 0):.1f}")
            print(f"  Easy Negatives: {final_stats.get('easy_negatives', 0):.1f}")
        
        print("="*60)
    
    def generate_all_plots(self, output_dir: str = "training_plots"):
        """Generate all visualization plots and save them."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Generating training visualizations in {output_path}/")
        
        # Generate individual plots
        self.plot_loss_curves(output_path / "loss_curves.png")
        self.plot_mining_statistics(output_path / "mining_statistics.png")
        self.plot_distance_metrics(output_path / "distance_metrics.png")
        self.plot_training_summary(output_path / "training_summary.png")
        
        # Print summary
        self.print_training_summary()
        
        print(f"\nAll plots saved to {output_path}/")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Visualize FaceNet training data")
    parser.add_argument("--history", "-f", default="checkpoints/training_history.json",
                       help="Path to training history JSON file")
    parser.add_argument("--output", "-o", default="training_plots",
                       help="Output directory for plots")
    parser.add_argument("--summary-only", "-s", action="store_true",
                       help="Only generate summary plot")
    
    args = parser.parse_args()
    
    try:
        visualizer = TrainingVisualizer(args.history)
        
        if args.summary_only:
            visualizer.plot_training_summary()
            visualizer.print_training_summary()
        else:
            visualizer.generate_all_plots(args.output)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 