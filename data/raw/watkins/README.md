# Watkins Marine Mammal Sound Database - Sperm Whale Samples

This directory contains sample sperm whale recordings from the
Watkins Marine Mammal Sound Database for testing and development.

## Source
- **Database**: Watkins Marine Mammal Sound Database
- **URL**: https://cis.whoi.edu/science/B/whalesounds/
- **Species**: Physeter macrocephalus (Sperm whale)
- **License**: Free for research use

## Files

- **watkins_62031001.wav** ✓
  - ID: 62031001
  - Description: Sperm whale clicks - Dominica recording

- **watkins_72009001.wav** ✓
  - ID: 72009001
  - Description: Sperm whale codas - 1972 recording

- **watkins_85004001.wav** ✓
  - ID: 85004001
  - Description: Sperm whale vocalizations - 1985 recording

- **watkins_91003001.wav** ✓
  - ID: 91003001
  - Description: Sperm whale communication - 1991 recording

- **watkins_91003005.wav** ✓
  - ID: 91003005
  - Description: Sperm whale social calls - 1991 recording

## Usage

These files can be used with the whale phonetic analyzer:

```python
from src.click_detector import ClickDetector
from src.coda_detector import CodaDetector

# Load and analyze a recording
detector = ClickDetector(sample_rate=44100)
audio, sr = detector.load_audio('data/raw/watkins/watkins_6569.wav')
clicks, envelope, threshold = detector.detect_clicks(audio)

# Group into codas
coda_detector = CodaDetector()
codas = coda_detector.group_clicks_to_codas(clicks)
```
