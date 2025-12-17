import os
import requests
import shutil
from pathlib import Path

def download_file(url, filename):
    """Download a file from a URL to a local path."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def main():
    # Create weights directory if it doesn't exist
    weights_dir = Path("weights/pre-trained-model")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # URL for the RF-DETR model weights
    # Note: This is a placeholder URL - you'll need to replace it with the actual URL
    model_url = "https://huggingface.co/spaces/IDEA-Research/RF-DETR/resolve/main/checkpoint_best_regular.pth"
    
    # Destination path
    dest_path = weights_dir / "checkpoint_best_regular.pth"
    
    print(f"Downloading RF-DETR model weights to {dest_path}...")
    try:
        download_file(model_url, dest_path)
        print("Download completed successfully!")
        print(f"File size: {dest_path.stat().st_size / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"Error downloading the model: {e}")
        print("\nPlease manually download the model weights from:")
        print("https://huggingface.co/spaces/IDEA-Research/RF-DETR/tree/main")
        print(f"And place them at: {dest_path}")

if __name__ == "__main__":
    main()
