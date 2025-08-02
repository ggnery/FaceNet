#!/usr/bin/env python3
"""
Download script for VGGFace2 and LFW datasets from Kaggle.

Usage:
    python download_dataset.py [--output-dir OUTPUT_DIR] [--extract] [--datasets DATASETS]
"""

import os
import argparse
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm


# Dataset configurations
DATASETS = {
    'vggface2': {
        'url': 'https://www.kaggle.com/api/v1/datasets/download/hearfool/vggface2',
        'filename': 'vggface2.zip',
        'extract_dir': 'vggface2'
    },
    'lfw': {
        'url': 'https://www.kaggle.com/api/v1/datasets/download/jessicali9530/lfw-dataset',
        'filename': 'lfw-dataset.zip',
        'extract_dir': 'lfw'
    }
}


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


def download_dataset(dataset_name: str, output_dir: Path, extract: bool = True) -> Path:
    """Download and optionally extract a specific dataset."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
    
    dataset_config = DATASETS[dataset_name]
    zip_path = output_dir / dataset_config['filename']
    
    print(f"\n=== Downloading {dataset_name.upper()} Dataset ===")
    
    # Download the dataset
    download_file(dataset_config['url'], zip_path)
    
    print(f"Download successful: {zip_path}")
    print(f"File size: {zip_path.stat().st_size / (1024**3):.2f} GB")
    
    # Extract if requested
    if extract:
        extract_dir = output_dir / dataset_config['extract_dir']
        extract_zip(zip_path, extract_dir)
        print(f"Dataset extracted to: {extract_dir}")
        return extract_dir
    else:
        print(f"To extract later, run:")
        print(f"python {__file__} --output-dir {output_dir} --datasets {dataset_name} --extract")
        return zip_path


def main():
    parser = argparse.ArgumentParser(description='Download VGGFace2 and/or LFW datasets')
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Directory to save the datasets (default: ./data)')
    parser.add_argument('--extract', action='store_true', default=True,
                        help='Extract the downloaded zip files (default: True)')
    parser.add_argument('--datasets', type=str, nargs='+', 
                        choices=['vggface2', 'lfw', 'all'], default=['all'],
                        help='Which datasets to download: vggface2, lfw, or all (default: all)')
    
    args = parser.parse_args()
    
    # Resolve output directory
    output_dir = Path(args.output_dir).expanduser().resolve()
    
    # Determine which datasets to download
    datasets_to_download = []
    if 'all' in args.datasets:
        datasets_to_download = ['vggface2', 'lfw']
    else:
        datasets_to_download = [d for d in args.datasets if d in DATASETS]
    
    print(f"Face Recognition Dataset Downloader")
    print(f"Output directory: {output_dir}")
    print(f"Datasets to download: {', '.join(datasets_to_download)}")
    
    downloaded_paths = []
    
    try:
        for dataset_name in datasets_to_download:
            result_path = download_dataset(dataset_name, output_dir, args.extract)
            downloaded_paths.append((dataset_name, result_path))
        
        print(f"\n=== Download Summary ===")
        for dataset_name, path in downloaded_paths:
            print(f"{dataset_name.upper()}: {path}")
        
        if args.extract:
            print(f"\nDatasets ready! You can now use these paths with train.py:")
            for dataset_name, path in downloaded_paths:
                print(f"python train.py --data_root {path}  # for {dataset_name}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. For Kaggle API: Ensure you have ~/.kaggle/kaggle.json with valid credentials")
        print("2. For direct download: Check your internet connection and URLs")
        print("3. Ensure you have sufficient disk space (VGGFace2 ~36GB, LFW ~180MB)")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 