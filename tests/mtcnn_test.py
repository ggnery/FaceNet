import sys
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import MTCNN

def main(args):
    # Check if files exist
    if not os.path.exists(args.checkpoints):
        print(f"Error: Checkpoints folder not found: {args.checkpoints}")
        return
    
    if not os.path.exists(args.image):
        print(f"Error: Test image not found: {args.image}")
        return
    
    print(f"Loading MTCNN with checkpoints from: {args.checkpoints}")
    print(f"Processing image: {args.image}")
    
    # Initialize MTCNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(
        pretrained_folder=args.checkpoints,
        device=device
    )
    
    # Load and process image
    img = Image.open(args.image).convert('RGB')
    
    print(f"Image size: {img.size}")
    print(f"Using device: {device}")
    
    # Detect faces
    print("Detecting faces...")
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
    
    if boxes is not None:
        print(f"Found {len(boxes)} face(s)")
        
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)
        
        # Draw bounding boxes and landmarks
        for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
            print(f"Face {i+1}: confidence = {prob:.4f}")
            print(f"  Bounding box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
            
            # Draw bounding box
            draw.rectangle(box.tolist(), outline='red', width=3)
            
            # Draw landmarks (eyes, nose, mouth)
            if landmark is not None:
                for point in landmark:
                    # Draw small circles for landmarks
                    x, y = point
                    draw.ellipse((x-2, y-2, x+2, y+2), fill='blue', outline='blue')
            
            # Add confidence text
            draw.text((box[0], box[1]-20), f'Face {i+1}: {prob:.3f}', 
                     fill='red')
        
        # Display using matplotlib
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_draw)
        plt.title(f'Detected Faces ({len(boxes)} found)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(args.save_output, dpi=150, bbox_inches='tight')
        print(f"Output saved to: {args.save_output}")
        plt.show()
        
    else:
        print("No faces detected in the image")
        
        # Show original image
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title('No Faces Detected')
        plt.axis('off')
        plt.savefig(args.save_output, dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MTCNN Face Detection Test')
    parser.add_argument('--checkpoints', required=True, 
                       help='Path to MTCNN checkpoints folder')
    parser.add_argument('--image', required=True,
                       help='Path to test image')
    parser.add_argument('--save_output', default='detected_faces.png',
                       help='Path to save output image with bounding boxes')
    
    args = parser.parse_args()
    main(args)