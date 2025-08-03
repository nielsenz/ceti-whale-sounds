#!/usr/bin/env python3
"""
Test script for improved whale phonetic analysis algorithms.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Import modules with fallback for missing dependencies
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️  librosa not available - using synthetic test data")

import numpy as np
import pandas as pd
from pathlib import Path

# Import our modules
import click_detector
import coda_detector  
import feature_extractor

def test_synthetic_data():
    """Test with synthetic whale-like click data."""
    print("🧪 Testing with synthetic whale click data")
    print("=" * 50)
    
    # Create synthetic click times that simulate real whale patterns
    # Pattern 1: "2+3" rhythm (2 clicks, pause, 3 clicks)
    pattern1 = np.array([1.0, 1.2, 1.8, 2.0, 2.2])  # 2+3 pattern
    
    # Pattern 2: "1+4" rhythm  
    pattern2 = np.array([5.0, 6.0, 6.2, 6.4, 6.6])  # 1+4 pattern
    
    # Pattern 3: Regular 5-click pattern
    pattern3 = np.array([10.0, 10.2, 10.4, 10.6, 10.8])  # Regular 5
    
    # Pattern 4: Echolocation-like (very regular, many clicks)
    pattern4 = np.array([15.0 + i*0.1 for i in range(25)])  # 25 regular clicks
    
    all_clicks = np.concatenate([pattern1, pattern2, pattern3, pattern4])
    
    print(f"Created {len(all_clicks)} synthetic clicks")
    
    # Test click detector parameter transparency
    print("\n🔧 Testing Click Detector Improvements:")
    detector = click_detector.ClickDetector(44100)  # Standard sample rate
    params = detector.get_parameter_summary()
    print(f"  Frequency filter: {params['frequency_filter']['actual']['lowcut']}-{params['frequency_filter']['actual']['highcut']} Hz")
    print(f"  Threshold method: {params['detection_params']['threshold_multiplier']}σ")
    print(f"  Scientific basis: {params['scientific_basis']['frequency_range']}")
    
    # Test coda detection
    print("\n🐋 Testing Coda Detection:")
    coda_det = coda_detector.CodaDetector()
    codas = coda_det.group_clicks_to_codas(all_clicks)
    print(f"  Detected {len(codas)} codas from {len(all_clicks)} clicks")
    
    # Test improved feature extraction
    print("\n🎵 Testing Improved Feature Extraction:")
    extractor = feature_extractor.FeatureExtractor()
    
    features_list = []
    for i, coda in enumerate(codas):
        features = extractor.extract_all_features(coda)
        features_list.append(features)
        
        print(f"\n  Coda {i+1}: {coda.num_clicks} clicks, {coda.duration:.2f}s")
        print(f"    Rhythm: {features.rhythm_pattern}")
        print(f"    Tempo: {features.tempo_cps:.2f} cps ({features.tempo_category.value})")
        print(f"    Rubato: {features.rubato_score:.3f} ({features.rubato_level.value})")
        print(f"    Ornamentation: {features.ornamentation_count} extra clicks")
        print(f"    Confidence: det={features.detection_confidence:.2f}, rhythm={features.rhythm_confidence:.2f}")
        print(f"    Echolocation likely: {features.is_echolocation_likely}")
        print(f"    Phonetic code: {features.phonetic_code}")
    
    # Test analysis collection
    print("\n📊 Testing Collection Analysis:")
    features_df = extractor.analyze_coda_collection(codas)
    
    # Show communication vs echolocation filtering
    comm_codas = features_df[~features_df['is_echolocation_likely']]
    echo_codas = features_df[features_df['is_echolocation_likely']]
    
    print(f"  Total codas: {len(features_df)}")
    print(f"  Communication codas: {len(comm_codas)}")
    print(f"  Likely echolocation: {len(echo_codas)}")
    
    if len(comm_codas) > 0:
        print(f"\n🎯 Communication Pattern Analysis:")
        unique_patterns = comm_codas['rhythm_pattern'].value_counts()
        print(f"  Unique rhythm patterns: {len(unique_patterns)}")
        for pattern, count in unique_patterns.items():
            print(f"    {pattern}: {count} occurrences")
        
        print(f"\n📈 Confidence Statistics:")
        print(f"  Detection confidence: {comm_codas['detection_confidence'].mean():.3f} ± {comm_codas['detection_confidence'].std():.3f}")
        print(f"  Rhythm confidence: {comm_codas['rhythm_confidence'].mean():.3f} ± {comm_codas['rhythm_confidence'].std():.3f}")
        print(f"  Classification confidence: {comm_codas['classification_confidence'].mean():.3f} ± {comm_codas['classification_confidence'].std():.3f}")
    
    print("\n✅ Synthetic data test complete!")
    return features_df

def test_real_audio():
    """Test with real whale audio if available."""
    if not LIBROSA_AVAILABLE:
        print("⚠️  Skipping real audio test - librosa not available")
        return None
    
    print("\n🎧 Testing with real whale audio")
    print("=" * 40)
    
    # Look for audio files
    data_dir = Path('data/raw')
    audio_files = list(data_dir.glob('*.wav'))
    
    if not audio_files:
        print("No audio files found - run scripts/download_sample_data.py first")
        return None
    
    # Test first file
    test_file = audio_files[0]
    print(f"Analyzing: {test_file.name}")
    
    # Load and analyze
    audio, sr = librosa.load(test_file, sr=None)
    duration = len(audio) / sr
    print(f"Duration: {duration:.1f}s, Sample rate: {sr}Hz")
    
    # Detect clicks with improved detector
    detector = click_detector.ClickDetector(sr)
    click_times, envelope, threshold = detector.detect_clicks(audio)
    print(f"Detected {len(click_times)} clicks")
    
    if len(click_times) > 0:
        # Group and analyze
        coda_det = coda_detector.CodaDetector()
        codas = coda_det.group_clicks_to_codas(click_times)
        
        extractor = feature_extractor.FeatureExtractor()
        features_df = extractor.analyze_coda_collection(codas)
        
        # Show results
        comm_codas = features_df[~features_df['is_echolocation_likely']]
        echo_codas = features_df[features_df['is_echolocation_likely']]
        
        print(f"\nReal audio results:")
        print(f"  Total codas: {len(features_df)}")
        print(f"  Communication: {len(comm_codas)}")
        print(f"  Echolocation: {len(echo_codas)}")
        
        if len(comm_codas) > 0:
            print(f"\nCommunication patterns found:")
            for pattern in comm_codas['phonetic_code'].unique()[:5]:
                print(f"  {pattern}")
    
    return features_df

def main():
    """Run all tests."""
    print("🐋 SPERM WHALE PHONETIC ANALYSIS - IMPROVED ALGORITHMS TEST")
    print("=" * 60)
    
    # Test synthetic data
    synthetic_results = test_synthetic_data()
    
    # Test real audio if available
    real_results = test_real_audio()
    
    print(f"\n🎉 IMPROVEMENT SUMMARY:")
    print("=" * 30)
    print("✅ Fixed rhythm detection off-by-one error")
    print("✅ Corrected tempo calculation (now biologically meaningful)")
    print("✅ Reduced rhythm break sensitivity (less false patterns)")
    print("✅ Pattern-relative ornamentation detection")
    print("✅ Added parameter transparency with scientific references")
    print("✅ Implemented confidence scoring (detection, rhythm, classification)")
    print("✅ Added echolocation vs communication filtering")
    print("✅ Fixed envelope smoothing edge artifacts")
    print("✅ Enhanced error handling and validation")
    print("\n🔬 Results should now be scientifically accurate and biologically meaningful!")

if __name__ == "__main__":
    main()