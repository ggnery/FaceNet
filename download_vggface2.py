#!/usr/bin/env python3
"""
Download script for VGGFace2 dataset from Kaggle.

Usage:
    python download_dataset.py [--output-dir OUTPUT_DIR] [--extract] 
"""

import os
import argparse
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress bar."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get total file size if available
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))
    
    print(f"Download completed: {output_path}")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract zip file with progress bar."""
    print(f"Extracting {zip_path} to {extract_to}")
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        
        with tqdm(desc="Extracting", total=len(file_list), unit='files') as progress_bar:
            for file_name in file_list:
                zip_ref.extract(file_name, extract_to)
                progress_bar.update(1)
    
    print(f"Extraction completed to: {extract_to}")



def download_with_requests(url: str, output_path: Path) -> Path:
    """Download dataset using direct HTTP request."""
    download_file(url, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Download VGGFace2 dataset')
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Directory to save the dataset (default: ./data)')
    parser.add_argument('--extract', action='store_true', default=True,
                        help='Extract the downloaded zip file')
    parser.add_argument('--url', type=str, 
                        default='https://www.kaggle.com/api/v1/datasets/download/hearfool/vggface2',    
                        help='Direct download URL (used when not using Kaggle API)')
    
    args = parser.parse_args()
    
    # Resolve output directory
    output_dir = Path(args.output_dir).expanduser().resolve()
    zip_filename = 'vggface2.zip'
    zip_path = output_dir / zip_filename
    
    print(f"VGGFace2 Dataset Downloader")
    print(f"Output directory: {output_dir}")
    
    try:
        # Use direct HTTP download
        downloaded_zip = download_with_requests(args.url, zip_path)
        
        print(f"Download successful: {downloaded_zip}")
        print(f"File size: {downloaded_zip.stat().st_size / (1024**3):.2f} GB")
        
        if args.extract:
            extract_dir = output_dir / 'vggface2'
            extract_zip(downloaded_zip, extract_dir)
            
            print(f"\nDataset ready at: {extract_dir}")
            print("You can now use this path with train.py:")
            print(f"python train.py --data_root {extract_dir}")
        else:
            print(f"\nTo extract the dataset, run:")
            print(f"python {__file__} --output-dir {output_dir} --extract")
            print(f"Or manually extract: {downloaded_zip}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. For Kaggle API: Ensure you have ~/.kaggle/kaggle.json with valid credentials")
        print("2. For direct download: Check your internet connection and URL")
        print("3. Ensure you have sufficient disk space (VGGFace2 is ~36GB)")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 