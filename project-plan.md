# Sperm Whale Phonetic Alphabet Explorer

## Complete Project Plan for Junior Developers

-----

## 🌊 Project Overview

### What You're Building

An open-source Python tool that helps marine biologists explore how sperm whales "talk" to each other. In May 2024, scientists discovered that sperm whale clicks follow patterns similar to an alphabet - with 143+ different combinations of rhythm, speed, timing variations, and decorative flourishes.

### Why This Project Matters

- **Scientific Impact**: No user-friendly tool exists yet for exploring this discovery
- **Conservation**: Understanding whale communication helps protect them
- **Career Building**: Great portfolio project combining audio processing, data science, and web development
- **Learning Opportunity**: Work with real scientific data while building practical skills

### Expected Timeline

- **4 weeks** for basic version
- **2-3 hours/day** commitment
- **Portfolio-ready** in 1 month

-----

## 📚 Background Knowledge

### The Scientific Discovery (Simplified)

Imagine whales have their own "alphabet" made of clicking sounds. Scientists found that whales combine 4 different elements to create "letters":

1. **Rhythm**: Pattern of clicks (like "click-pause-click-click-click")
1. **Tempo**: How fast they click overall
1. **Rubato**: Small timing variations (like a jazz musician)
1. **Ornamentation**: Extra clicks added for emphasis

### What You Need to Know

- **Python Basics**: Functions, classes, lists, dictionaries
- **Audio Concepts**: Sound is just numbers over time (samples)
- **No Prior Experience Needed In**: Marine biology, advanced signal processing, or machine learning

-----

## 🗺️ Complete Roadmap

### Phase 1: Foundation (Week 1)

**Goal**: Understand the data and detect whale clicks

### Phase 2: Feature Extraction (Week 2)

**Goal**: Extract the 4 phonetic components from clicks

### Phase 3: User Interface (Week 3)

**Goal**: Build an interactive tool researchers can actually use

### Phase 4: Polish & Share (Week 4)

**Goal**: Test, document, and share with the scientific community

-----

## 📊 Getting the Data

### Primary Source: Watkins Marine Mammal Sound Database

**What It Is**:

- Free online database with 60+ years of whale recordings
- Created by pioneers in whale research
- Professionally verified recordings with metadata

**How to Access**:

1. Go to: https://cis.whoi.edu/science/B/whalesounds/
1. Click "Best of" cuts tab
1. Search for "Physeter macrocephalus" (scientific name for sperm whale)
1. Download these specific files (good examples):
- **File 6569**: Multiple clear codas (communication patterns)
- **File 6571**: Echolocation clicks (for hunting)
- **File 6572**: Social vocalizations
- **File 1036**: Atlantic Ocean recording
- **File 1419**: Pacific Ocean recording

**Why Start Here**:

- High-quality recordings
- Free and legal to use
- Good variety of click patterns
- Includes recording context

### Data Organization Structure

```
whale-phonetics-project/
├── data/
│   ├── raw/
│   │   ├── watkins/         # Original downloads
│   │   └── other/           # Future datasets
│   ├── processed/           # Your analysis results
│   └── annotations/         # Labeled clicks/codas
├── src/                     # Your code
├── notebooks/               # Jupyter experiments
├── results/                 # Outputs and figures
└── docs/                    # Documentation
```

### Expected Data Volume

- **Initial**: 5-10 recordings (enough to start)
- **Total Available**: ~50-100 sperm whale recordings
- **Each Recording**: 30 seconds to 5 minutes
- **File Size**: ~5-50 MB per recording

-----

## 🛠️ Technical Methodology

### Core Approach: Digital Signal Processing Pipeline

```
Audio File → Click Detection → Coda Grouping → Feature Extraction → Database → Visualization
```

### Step 1: Click Detection

**What**: Find individual clicking sounds in continuous audio

**Method**: Energy-based detection with bandpass filtering

```python
# Pseudocode
1. Filter audio to whale click frequencies (2-20 kHz)
2. Calculate signal energy over time
3. Find peaks above threshold
4. Mark peak locations as clicks
```

**Why This Works**:

- Whale clicks are loud and distinct
- They occur in specific frequency range
- Energy peaks clearly mark click times

### Step 2: Coda Grouping

**What**: Group individual clicks into communication units (codas)

**Method**: Inter-click interval (ICI) analysis

```python
# If clicks are close together (< 2 seconds), they're part of same coda
# If gap is large (> 2 seconds), new coda starts
```

**Why This Works**:

- Whales pause between different "words" (codas)
- Clicks within a coda are related
- Similar to how we group letters into words

### Step 3: Feature Extraction

**What**: Measure the 4 phonetic components

**Methods**:

1. **Rhythm**: Count clicks between long pauses
1. **Tempo**: Calculate clicks per second
1. **Rubato**: Measure timing irregularity
1. **Ornamentation**: Detect extra clicks

**Why These Features**:

- Matches the scientific paper's findings
- Captures how whales vary their communication
- Can be calculated objectively

### Step 4: Pattern Analysis

**What**: Find and categorize different phonetic combinations

**Method**: Database queries and clustering

```python
# Each coda gets a "phonetic code" like:
# "1+3_fast_moderate_ornament" 
# Meaning: 1+3 rhythm, fast tempo, moderate rubato, with ornament
```

-----

## 💻 Development Environment Setup

### Required Software

```bash
# 1. Install Anaconda (Python package manager)
# Download from: https://www.anaconda.com/products/individual

# 2. Create project environment
conda create -n whale-phonetics python=3.10
conda activate whale-phonetics

# 3. Install required packages
pip install numpy          # Numerical computing
pip install scipy          # Signal processing  
pip install librosa        # Audio analysis
pip install pandas         # Data organization
pip install matplotlib     # Basic plotting
pip install plotly         # Interactive plots
pip install streamlit      # Web interface
pip install jupyter        # Interactive notebooks
```

### Why These Libraries?

- **NumPy/SciPy**: Foundation for numerical work
- **Librosa**: Specialized for audio (used by music/speech researchers)
- **Pandas**: Organizes data like Excel but programmable
- **Streamlit**: Creates web apps with just Python (no HTML/CSS needed)

-----

## 📝 Week-by-Week Implementation

## Week 1: Foundation - Click Detection

### Day 1-2: First Audio Exploration

**Objective**: Load and visualize whale sounds

**Tutorial Notebook**: `01_explore_audio.ipynb`

```python
import numpy as np
import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd

# Load your first whale recording
audio_file = 'data/raw/watkins/watkins_6569.wav'
audio_data, sample_rate = librosa.load(audio_file, sr=None)

print(f"Loaded {len(audio_data)} samples at {sample_rate} Hz")
print(f"Duration: {len(audio_data)/sample_rate:.2f} seconds")

# Visualize the sound
plt.figure(figsize=(12, 6))

# Top: Waveform (amplitude over time)
plt.subplot(2, 1, 1)
time_axis = np.arange(len(audio_data)) / sample_rate
plt.plot(time_axis, audio_data)
plt.ylabel('Amplitude')
plt.title('Whale Recording Waveform')

# Bottom: Spectrogram (frequencies over time)
plt.subplot(2, 1, 2)
D = librosa.stft(audio_data)  # Short-time Fourier transform
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
librosa.display.specshow(S_db, sr=sample_rate, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.ylim(0, 20000)  # Whale clicks are in 2-20kHz range

plt.tight_layout()
plt.show()

# Listen to the audio
ipd.Audio(audio_data, rate=sample_rate)
```

**What You Should See**:

- Waveform: Spikes where clicks occur
- Spectrogram: Vertical lines at click times
- Audio: Clicking sounds (use headphones!)

**Learning Goals**:

- ✅ Understand audio as numerical data
- ✅ Visualize sound in time and frequency
- ✅ Identify clicks visually

### Day 3-4: Automated Click Detection

**Objective**: Write code to find clicks automatically

**Implementation**: `src/click_detector.py`

```python
import numpy as np
from scipy import signal

class SimpleClickDetector:
    """
    Detects sperm whale clicks in audio recordings.
    
    How it works:
    1. Filters audio to whale frequency range (removes noise)
    2. Calculates energy envelope (loudness over time)
    3. Finds peaks in energy (potential clicks)
    4. Returns click times
    """
    
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.lowcut = 2000   # Hz - whale clicks are above this
        self.highcut = 20000 # Hz - and below this
        
    def bandpass_filter(self, audio):
        """Remove frequencies outside whale click range"""
        nyquist = self.sample_rate / 2
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        # Create filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered = signal.filtfilt(b, a, audio)
        return filtered
    
    def get_energy_envelope(self, audio):
        """Calculate loudness over time"""
        # Hilbert transform gives us envelope
        analytic_signal = signal.hilbert(audio)
        envelope = np.abs(analytic_signal)
        
        # Smooth to avoid tiny peaks
        window = signal.windows.hann(int(0.001 * self.sample_rate))
        envelope_smooth = signal.convolve(envelope, window, mode='same')
        
        return envelope_smooth
    
    def find_click_peaks(self, envelope):
        """Find times where clicks occur"""
        # Dynamic threshold: mean + 3 * standard deviation
        threshold = np.mean(envelope) + 3 * np.std(envelope)
        
        # Find peaks
        min_distance = int(0.01 * self.sample_rate)  # Clicks at least 10ms apart
        peaks, _ = signal.find_peaks(envelope, 
                                     height=threshold,
                                     distance=min_distance)
        
        # Convert to seconds
        click_times = peaks / self.sample_rate
        
        return click_times, threshold
    
    def detect_clicks(self, audio):
        """Main function: audio in, click times out"""
        # Step 1: Filter
        filtered = self.bandpass_filter(audio)
        
        # Step 2: Get envelope
        envelope = self.get_energy_envelope(filtered)
        
        # Step 3: Find peaks
        clicks, threshold = self.find_click_peaks(envelope)
        
        return clicks, envelope, threshold
```

**Testing Your Detector**:

```python
# Test on a recording
detector = SimpleClickDetector(sample_rate)
clicks, envelope, threshold = detector.detect_clicks(audio_data)

print(f"Found {len(clicks)} clicks")
print(f"First 5 click times: {clicks[:5]}")

# Visualize detection
plt.figure(figsize=(15, 5))
plt.plot(time_axis, audio_data, alpha=0.5, label='Audio')
plt.plot(time_axis, envelope, label='Energy')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.scatter(clicks, np.ones_like(clicks) * threshold, 
           color='red', s=100, marker='v', label='Detected clicks')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Click Detection Results')
plt.show()
```

### Day 5: Grouping Clicks into Codas

**Objective**: Organize clicks into communication units

**Why**: Individual clicks are like letters; codas are like words

**Implementation**: `src/coda_detector.py`

```python
class CodaDetector:
    """
    Groups clicks into codas (communication units).
    
    A coda is a sequence of clicks that form a pattern,
    like "click-click-pause-click-click-click"
    """
    
    def __init__(self, max_ici=2.0, min_clicks=3):
        """
        max_ici: Maximum seconds between clicks in same coda
        min_clicks: Minimum clicks to count as a coda
        """
        self.max_ici = max_ici
        self.min_clicks = min_clicks
    
    def group_clicks_to_codas(self, click_times):
        """Group clicks based on time gaps"""
        if len(click_times) < self.min_clicks:
            return []
        
        codas = []
        current_coda = [click_times[0]]
        
        for i in range(1, len(click_times)):
            time_gap = click_times[i] - click_times[i-1]
            
            if time_gap <= self.max_ici:
                # Still same coda
                current_coda.append(click_times[i])
            else:
                # New coda starting
                if len(current_coda) >= self.min_clicks:
                    codas.append(np.array(current_coda))
                current_coda = [click_times[i]]
        
        # Don't forget last coda
        if len(current_coda) >= self.min_clicks:
            codas.append(np.array(current_coda))
        
        return codas
    
    def analyze_coda(self, coda_clicks):
        """Get basic stats about a coda"""
        stats = {
            'num_clicks': len(coda_clicks),
            'duration': coda_clicks[-1] - coda_clicks[0],
            'mean_ici': np.mean(np.diff(coda_clicks)) if len(coda_clicks) > 1 else 0,
            'start_time': coda_clicks[0],
            'end_time': coda_clicks[-1]
        }
        return stats
```

**Week 1 Milestone Checklist**:

- [ ] Downloaded 5+ whale recordings
- [ ] Can load and visualize audio
- [ ] Click detector finds 80%+ of visible clicks
- [ ] Grouped clicks into 20+ codas
- [ ] Understand each step of the process

-----

## Week 2: Extracting the Four Phonetic Components

### Day 6-7: Rhythm Detection

**Objective**: Classify patterns like "1+3" or "2+3"

**Method**: Find groups based on pause duration

```python
class RhythmAnalyzer:
    """
    Determines rhythm patterns from click timing.
    
    Example: clicks at [0, 0.2, 0.8, 1.0, 1.2] seconds
    Small gaps: 0.2s between first two, 0.2s between last three
    Large gap: 0.6s in middle
    Pattern: "2+3" (2 clicks, pause, 3 clicks)
    """
    
    def extract_rhythm(self, coda_clicks):
        if len(coda_clicks) < 2:
            return "single"
            
        # Calculate inter-click intervals
        icis = np.diff(coda_clicks)
        
        # Find rhythm breaks (gaps 50% larger than average)
        mean_ici = np.mean(icis)
        rhythm_breaks = []
        
        for i, ici in enumerate(icis):
            if ici > 1.5 * mean_ici:
                rhythm_breaks.append(i)
        
        # Count clicks in each group
        groups = []
        start = 0
        
        for break_pos in rhythm_breaks:
            group_size = break_pos - start + 1
            groups.append(group_size)
            start = break_pos + 1
            
        # Don't forget last group
        groups.append(len(coda_clicks) - start)
        
        # Create rhythm string (e.g., "1+3")
        rhythm = "+".join(map(str, groups))
        
        return rhythm
```

### Day 8-9: Tempo Analysis

**Objective**: Measure overall speed

```python
def calculate_tempo(coda_clicks):
    """
    Calculate clicks per second.
    Like measuring words per minute in speech.
    """
    if len(coda_clicks) < 2:
        return 0, "single"
        
    duration = coda_clicks[-1] - coda_clicks[0]
    clicks_per_second = (len(coda_clicks) - 1) / duration
    
    # Categorize
    if clicks_per_second < 2:
        category = "slow"
    elif clicks_per_second < 4:
        category = "medium"
    else:
        category = "fast"
        
    return clicks_per_second, category
```

### Day 10: Rubato and Ornamentation

**Objective**: Detect timing variations and decorative clicks

```python
def detect_rubato(coda_clicks):
    """
    Measure how "jazzy" the timing is.
    Regular = low rubato, Irregular = high rubato
    """
    icis = np.diff(coda_clicks)
    expected = np.mean(icis)
    deviations = icis - expected
    rubato_score = np.std(deviations) / expected
    return rubato_score

def detect_ornamentation(coda_clicks, typical_length=5):
    """
    Check for extra clicks beyond normal pattern
    """
    has_extra = len(coda_clicks) > typical_length
    extra_clicks = max(0, len(coda_clicks) - typical_length)
    return has_extra, extra_clicks
```

-----

## Week 3: Building the Interactive Explorer

### Day 11-13: Streamlit Web Interface

**Why Streamlit**: Makes web apps with pure Python - no HTML/CSS needed!

**Basic App Structure**: `app.py`

```python
import streamlit as st
import pandas as pd

st.title("🐋 Sperm Whale Phonetic Explorer")

# Sidebar for filters
st.sidebar.header("Filter Options")
rhythm_filter = st.sidebar.selectbox("Rhythm Pattern", 
                                    ["All", "1+3", "2+3", "1+4"])

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Coda Patterns")
    # Display patterns here
    
with col2:
    st.subheader("Audio Player")
    # Audio playback here

# Run with: streamlit run app.py
```

### Day 14-15: Final Integration

Connect all components into working system.

-----

## Week 4: Testing and Sharing

### Testing Strategy

1. **Unit Tests**: Test each function individually
1. **Integration Tests**: Test full pipeline
1. **User Testing**: Have someone else try your tool

### Documentation Essentials

- **README.md**: How to install and use
- **Examples**: Jupyter notebook with tutorial
- **Comments**: Explain complex code sections

-----

## 🎯 Definition of Success

### Minimum Viable Product (MVP)

- [ ] Processes 5+ recordings successfully
- [ ] Detects 50+ codas total
- [ ] Extracts all 4 phonetic features
- [ ] Web interface displays patterns
- [ ] Can search/filter codas
- [ ] Plays audio for each coda

### Stretch Goals

- [ ] Pattern discovery (find new combinations)
- [ ] Export results to research formats
- [ ] Process hour-long recordings
- [ ] Real-time analysis

-----

## 🤝 Getting Help

### When You're Stuck

1. **Audio Processing Issues**
- Search: "librosa [your specific issue]"
- Common problems: sample rate mismatches, file formats
1. **Signal Processing**
- Search: "scipy signal processing [technique]"
- Stack Overflow has great DSP answers
1. **Whale-Specific Questions**
- Bioacoustics Stack Exchange
- Read the Nature Communications paper (even if parts are complex)
1. **General Python**
- Start with official documentation
- /r/learnpython is beginner-friendly

### Code Review Checklist

Before sharing your code:

- [ ] Remove hardcoded paths
- [ ] Add docstrings to functions
- [ ] Create requirements.txt
- [ ] Test on a clean environment
- [ ] Include sample data or download instructions

-----

## 🚀 Next Steps After Completion

### 1. Share Your Work

- Upload to GitHub with clear README
- Post demo video on LinkedIn/Twitter
- Write a blog post about what you learned

### 2. Get Feedback

- Email researchers (template provided in main document)
- Share in bioacoustics communities
- Ask for code review from experienced developers

### 3. Extend the Project

- Add more species (dolphins, humpback whales)
- Implement real-time analysis
- Create mobile app for field recording
- Contribute to existing bioacoustics projects

### 4. Career Applications

This project demonstrates:

- **Signal Processing**: Valuable in many fields
- **Data Analysis**: Universal skill
- **Scientific Computing**: Growing field
- **Web Development**: Practical application
- **Open Source Contribution**: Great for resume

-----

## 📚 Additional Resources

### Papers to Read (Start with Abstracts)

1. Sharma et al. (2024) - The phonetic alphabet paper
1. Watkins & Schevill historic papers - Pioneers in the field

### Online Courses

- "Audio Signal Processing for Music Applications" (Coursera)
- "Python for Data Science" (EdX)

### Communities

- Bioacoustics Stack Exchange
- PyData community
- Women in Machine Learning (WiML)

### Tools to Explore Later

- Raven Pro - Cornell's bioacoustics software
- Audacity - Audio editing (good for understanding)
- Google Colab - Free cloud computing for experiments

-----

## 💡 Final Tips

### For Success

1. **Start Small**: Get one recording working perfectly first
1. **Visualize Everything**: Plots help understanding
1. **Save Incrementally**: Version control from day 1
1. **Ask Questions**: No question is too basic
1. **Document as You Go**: Future you will thank present you

### Common Pitfalls to Avoid

1. **Trying to process all data at once** - Start with one file
1. **Perfectionism** - Working code > perfect code
1. **Not testing edge cases** - What if there are no clicks?
1. **Skipping visualization** - Always plot your results
1. **Working in isolation** - Share early, get feedback

### Remember

You're building something that doesn't exist yet. You're contributing to science. You're learning valuable skills. Every error is a learning opportunity. The whales are counting on us to understand them better!

-----

## 🐋 Welcome to the Deep End!

You're about to embark on an exciting journey combining code, science, and conservation. Take it one day at a time, celebrate small victories, and remember - you're building a tool that could help us understand one of Earth's most intelligent species.

Happy coding! 🌊