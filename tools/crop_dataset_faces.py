import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import MTCNN
from model.mtcnn.detect_face import extract_face
import argparse
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_image_paths(root_dir):
    """Recursively get all image file paths from the root directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_paths = []
    
    for path in Path(root_dir).rglob('*'):
        if path.is_file() and path.suffix.lower() in image_extensions:
            image_paths.append(path)
    
    return image_paths

def create_output_structure(input_path, root_input_dir, root_output_dir):
    """Create the corresponding output directory structure."""
    relative_path = input_path.relative_to(root_input_dir)
    
    output_path = Path(root_output_dir) / relative_path
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    return output_path

def process_image(image_path, model, root_input_dir, root_output_dir):
    """Process a single image: detect face, crop, and save."""
    try:
        image = Image.open(image_path).convert('RGB')
        boxes, probs = model.detect(image)
        
        output_path = create_output_structure(image_path, root_input_dir, root_output_dir)
        
        if boxes is not None and len(boxes) > 0:
            # Find the box with highest confidence
            best_idx = np.argmax(probs)
            best_box = boxes[best_idx]
            
            _ = extract_face(image, best_box, save_path=output_path)
            
            return True, f"Face detected and cropped for {image_path.name}"
        else:
            # No face detected, copy the full image
            shutil.copy2(image_path, output_path)
            return True, f"No face detected, copied full image for {image_path.name}"
            
    except Exception as e:
        return False, f"Error processing {image_path.name}: {str(e)}"

def main(args):
    model = MTCNN(pretrained_folder=args.mtcnn_weights, device=device)   
    
    root_dir = Path(args.data_path)
    output_dir = Path(args.output_dir)
    
    if not root_dir.exists():
        print(f"Error: Input directory {root_dir} does not exist!")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
     
    print("Scanning for images...")
    image_paths = get_image_paths(root_dir)
    
    if not image_paths:
        print("No images found in the specified directory!")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process each image
    success_count = 0
    error_count = 0
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        success, message = process_image(image_path, model, root_dir, output_dir)
        
        if success:
            success_count += 1
        else:
            error_count += 1
            print(f"Error: {message}")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count} images")
    print(f"Errors: {error_count} images")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop VGGFace2 and/or LFW dataset faces')
    parser.add_argument('--output_dir', type=str, default='./data/croped', help='Directory to save the copied cropped datasets (default: ./data/croped)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset that will be cropped')
    parser.add_argument('--mtcnn_weights', default='./checkpoints/mtcnn', help='Path to MTCNN pretrained weights folder')
     
    args = parser.parse_args()
    main(args)