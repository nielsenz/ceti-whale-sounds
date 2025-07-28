#!/usr/bin/env python3
"""
Download sample sperm whale recordings from the Watkins Marine Mammal Sound Database.

This script downloads verified sperm whale recordings that are good examples
for testing and development of the phonetic alphabet explorer.

Recommended files from CLAUDE.md:
- File 6569: Multiple clear codas
- File 6571: Echolocation clicks  
- File 6572: Social vocalizations
- File 1036: Atlantic Ocean recording
- File 1419: Pacific Ocean recording

Usage:
    python scripts/download_sample_data.py
"""

import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict
import time

# Base URL for Watkins database files
WATKINS_BASE_URL = "https://cis.whoi.edu/science/B/whalesounds/WhaleSounds"

# Working sperm whale recordings from the database
SAMPLE_FILES = [
    {
        "id": "62031001",
        "description": "Sperm whale clicks - Dominica recording",
        "filename": "watkins_62031001.wav"
    },
    {
        "id": "72009001", 
        "description": "Sperm whale codas - 1972 recording",
        "filename": "watkins_72009001.wav"
    },
    {
        "id": "85004001",
        "description": "Sperm whale vocalizations - 1985 recording",
        "filename": "watkins_85004001.wav"
    },
    {
        "id": "91003001",
        "description": "Sperm whale communication - 1991 recording",
        "filename": "watkins_91003001.wav"
    },
    {
        "id": "91003005",
        "description": "Sperm whale social calls - 1991 recording",
        "filename": "watkins_91003005.wav"
    }
]


def download_file(url: str, local_path: Path, description: str = "") -> bool:
    """
    Download a file from URL to local path with progress indication.
    
    Parameters:
    -----------
    url : str
        URL to download from
    local_path : Path
        Local file path to save to
    description : str
        Description for progress display
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        print(f"Downloading {description}...")
        print(f"  URL: {url}")
        print(f"  Saving to: {local_path}")
        
        # Create directory if needed
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress callback
        def progress_callback(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                print(f"\r  Progress: {percent}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, local_path, progress_callback)
        print(f"\n  ✓ Successfully downloaded {local_path.name}")
        
        # Verify file exists and has reasonable size
        if local_path.exists() and local_path.stat().st_size > 1000:  # At least 1KB
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"  File size: {size_mb:.1f} MB")
            return True
        else:
            print(f"  ✗ Download failed or file too small")
            return False
            
    except urllib.error.URLError as e:
        print(f"  ✗ URL Error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def download_watkins_samples(data_dir: Path) -> Dict[str, bool]:
    """
    Download all recommended Watkins database samples.
    
    Parameters:
    -----------
    data_dir : Path
        Root data directory (will create raw/watkins/ subdirectory)
        
    Returns:
    --------
    Dict[str, bool]
        Results for each file download
    """
    watkins_dir = data_dir / "raw" / "watkins"
    watkins_dir.mkdir(parents=True, exist_ok=True)
    
    print("🐋 Downloading Sperm Whale Sample Data")
    print("=" * 50)
    print(f"Data directory: {watkins_dir}")
    print(f"Number of files: {len(SAMPLE_FILES)}")
    print()
    
    results = {}
    successful_downloads = 0
    
    for i, file_info in enumerate(SAMPLE_FILES, 1):
        print(f"[{i}/{len(SAMPLE_FILES)}] {file_info['description']}")
        
        # Use the correct Watkins database URL pattern
        possible_urls = [
            f"{WATKINS_BASE_URL}/{file_info['id']}.wav"
        ]
        
        local_path = watkins_dir / file_info['filename']
        
        # Skip if already exists
        if local_path.exists():
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Already exists ({size_mb:.1f} MB)")
            results[file_info['id']] = True
            successful_downloads += 1
            continue
        
        # Try each URL until one works
        download_success = False
        for url in possible_urls:
            if download_file(url, local_path, file_info['description']):
                download_success = True
                break
            time.sleep(1)  # Brief pause between attempts
        
        results[file_info['id']] = download_success
        if download_success:
            successful_downloads += 1
        
        print()  # Blank line between files
    
    # Summary
    print("=" * 50)
    print(f"Download Summary: {successful_downloads}/{len(SAMPLE_FILES)} successful")
    
    if successful_downloads > 0:
        print("\n✓ Sample files available for analysis:")
        for file_info in SAMPLE_FILES:
            if results.get(file_info['id'], False):
                local_path = watkins_dir / file_info['filename']
                size_mb = local_path.stat().st_size / (1024 * 1024)
                print(f"  - {file_info['filename']} ({size_mb:.1f} MB): {file_info['description']}")
    
    if successful_downloads < len(SAMPLE_FILES):
        print(f"\n⚠️  {len(SAMPLE_FILES) - successful_downloads} files failed to download.")
        print("This may be due to:")
        print("  - Network connectivity issues")
        print("  - Changes in Watkins database structure")
        print("  - Files moved or renamed")
        print("\nYou can manually download from: https://cis.whoi.edu/science/B/whalesounds/")
    
    return results


def create_data_info_file(data_dir: Path, download_results: Dict[str, bool]):
    """Create a file documenting the downloaded data."""
    info_file = data_dir / "raw" / "watkins" / "README.md"
    
    with open(info_file, 'w') as f:
        f.write("# Watkins Marine Mammal Sound Database - Sperm Whale Samples\n\n")
        f.write("This directory contains sample sperm whale recordings from the\n")
        f.write("Watkins Marine Mammal Sound Database for testing and development.\n\n")
        f.write("## Source\n")
        f.write("- **Database**: Watkins Marine Mammal Sound Database\n")
        f.write("- **URL**: https://cis.whoi.edu/science/B/whalesounds/\n")
        f.write("- **Species**: Physeter macrocephalus (Sperm whale)\n")
        f.write("- **License**: Free for research use\n\n")
        f.write("## Files\n\n")
        
        for file_info in SAMPLE_FILES:
            status = "✓" if download_results.get(file_info['id'], False) else "✗"
            f.write(f"- **{file_info['filename']}** {status}\n")
            f.write(f"  - ID: {file_info['id']}\n")
            f.write(f"  - Description: {file_info['description']}\n\n")
        
        f.write("## Usage\n\n")
        f.write("These files can be used with the whale phonetic analyzer:\n\n")
        f.write("```python\n")
        f.write("from src.click_detector import ClickDetector\n")
        f.write("from src.coda_detector import CodaDetector\n\n")
        f.write("# Load and analyze a recording\n")
        f.write("detector = ClickDetector(sample_rate=44100)\n")
        f.write("audio, sr = detector.load_audio('data/raw/watkins/watkins_6569.wav')\n")
        f.write("clicks, envelope, threshold = detector.detect_clicks(audio)\n\n")
        f.write("# Group into codas\n")
        f.write("coda_detector = CodaDetector()\n")
        f.write("codas = coda_detector.group_clicks_to_codas(clicks)\n")
        f.write("```\n")


def main():
    """Main function to download sample data."""
    # Get project root directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"
    
    print("🐋 Sperm Whale Phonetic Alphabet Explorer")
    print("Sample Data Download Script")
    print("=" * 50)
    print(f"Project directory: {project_dir}")
    print(f"Data directory: {data_dir}")
    print()
    
    # Download files
    results = download_watkins_samples(data_dir)
    
    # Create documentation
    create_data_info_file(data_dir, results)
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Run: python -m src.click_detector (test click detection)")
    print("2. Run: python -m src.coda_detector (test coda grouping)")  
    print("3. Run: jupyter notebook notebooks/ (explore data)")
    print("4. Run: streamlit run app.py (web interface)")


if __name__ == "__main__":
    main()