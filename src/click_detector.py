"""
Sperm whale click detection using energy-based analysis.

This module detects individual whale clicks in audio recordings using:
1. Bandpass filtering (2-20 kHz) to isolate whale click frequencies
2. Energy envelope calculation to track signal loudness over time
3. Peak detection with adaptive thresholding to find click times

The approach is based on the fact that sperm whale clicks are:
- High-energy, short-duration pulses
- Concentrated in the 2-20 kHz frequency range
- Distinct from background ocean noise
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional, Dict
import soundfile as sf
import warnings


class ClickDetector:
    """
    Detects sperm whale clicks in audio recordings using energy-based analysis.
    
    This detector works by:
    1. Filtering audio to whale click frequencies (2-20 kHz)
    2. Computing an energy envelope to track signal power over time
    3. Finding peaks in the energy that exceed a dynamic threshold
    4. Returning click times in seconds
    
    Parameters:
    -----------
    sample_rate : int
        Audio sample rate in Hz
    lowcut : float, default=2000
        Lower frequency bound for whale clicks (Hz)
    highcut : float, default=20000
        Upper frequency bound for whale clicks (Hz)
    threshold_multiplier : float, default=3.0
        Multiplier for dynamic threshold (mean + multiplier * std)
    min_click_separation : float, default=0.01
        Minimum time between clicks in seconds (prevents double-detection)
    """
    
    def __init__(
        self, 
        sample_rate: int,
        lowcut: float = 2000,     # Gero et al. 2016: sperm whale clicks 2-20kHz
        highcut: float = 20000,   # Watkins 1985: peak energy 5-15kHz
        threshold_multiplier: float = 3.0,  # 3-sigma detection (99.7% confidence)
        min_click_separation: float = 0.01  # Madsen 2002: min ICI ~10ms
    ):
        self.sample_rate = sample_rate
        self.requested_lowcut = lowcut
        self.requested_highcut = highcut
        self.threshold_multiplier = threshold_multiplier
        self.min_click_separation = min_click_separation
        
        # Validate and adjust frequency bounds with transparency
        nyquist = sample_rate / 2
        if highcut >= nyquist:
            self.actual_highcut = nyquist * 0.95
            warnings.warn(
                f"Requested highcut {highcut}Hz exceeds Nyquist frequency. "
                f"Using {self.actual_highcut:.0f}Hz instead for {sample_rate}Hz audio. "
                f"This changes the analysis frequency range and may affect results.",
                UserWarning
            )
        else:
            self.actual_highcut = highcut
        
        self.actual_lowcut = lowcut
        
        # Store both requested and actual parameters for transparency
        self.filter_params = {
            'requested': {'lowcut': lowcut, 'highcut': highcut},
            'actual': {'lowcut': self.actual_lowcut, 'highcut': self.actual_highcut},
            'sample_rate': sample_rate,
            'nyquist': nyquist
        }
        
        # Validate other parameters
        self._validate_parameters()
    
    def bandpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to isolate whale click frequencies.
        
        Sperm whale clicks are concentrated in the 2-20 kHz range, so we filter
        out lower frequency ocean noise and higher frequency electronic noise.
        
        Parameters:
        -----------
        audio : np.ndarray
            Input audio signal
            
        Returns:
        --------
        np.ndarray
            Filtered audio signal
        """
        nyquist = self.sample_rate / 2
        low = self.actual_lowcut / nyquist
        high = self.actual_highcut / nyquist
        
        # Design 4th-order Butterworth bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter using filtfilt for zero-phase filtering
        filtered = signal.filtfilt(b, a, audio)
        
        return filtered
    
    def compute_energy_envelope(self, audio: np.ndarray) -> np.ndarray:
        """
        Calculate energy envelope using Hilbert transform.
        
        The energy envelope tracks the instantaneous power of the signal,
        which peaks strongly at whale click locations.
        
        Parameters:
        -----------
        audio : np.ndarray
            Filtered audio signal
            
        Returns:
        --------
        np.ndarray
            Energy envelope signal
        """
        # Hilbert transform gives analytic signal
        analytic_signal = signal.hilbert(audio)
        
        # Envelope is the magnitude of the analytic signal
        envelope = np.abs(analytic_signal)
        
        # Smooth the envelope to reduce noise
        # Window size: 1ms at current sample rate
        window_size = max(1, int(0.001 * self.sample_rate))
        if window_size > 1:
            window = signal.windows.hann(window_size)
            window = window / np.sum(window)  # Normalize
            
            # FIXED: Use reflection padding to avoid edge artifacts
            pad_width = (len(window) - 1) // 2
            padded_envelope = np.pad(envelope, pad_width, mode='reflect')
            smoothed = signal.convolve(padded_envelope, window, mode='valid')
            
            # Ensure output matches input length
            if len(smoothed) != len(envelope):
                # Adjust for any size mismatch
                if len(smoothed) > len(envelope):
                    smoothed = smoothed[:len(envelope)]
                else:
                    # Pad if too short (rare case)
                    smoothed = np.pad(smoothed, (0, len(envelope) - len(smoothed)), mode='edge')
            
            envelope = smoothed
        
        return envelope
    
    def find_click_peaks(self, envelope: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Find peaks in energy envelope that correspond to whale clicks.
        
        Uses adaptive thresholding based on signal statistics to handle
        varying noise levels and click amplitudes.
        
        Parameters:
        -----------
        envelope : np.ndarray
            Energy envelope signal
            
        Returns:
        --------
        click_samples : np.ndarray
            Sample indices of detected clicks
        threshold : float
            Threshold value used for detection
        """
        # Dynamic threshold based on signal statistics
        mean_energy = np.mean(envelope)
        std_energy = np.std(envelope)
        threshold = mean_energy + self.threshold_multiplier * std_energy
        
        # Minimum distance between peaks (in samples)
        min_distance = int(self.min_click_separation * self.sample_rate)
        
        # Find peaks above threshold
        peaks, properties = signal.find_peaks(
            envelope,
            height=threshold,
            distance=min_distance
        )
        
        return peaks, threshold
    
    def detect_clicks(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Main detection function: audio in, click times out.
        
        Parameters:
        -----------
        audio : np.ndarray
            Raw audio signal
            
        Returns:
        --------
        click_times : np.ndarray
            Times of detected clicks in seconds
        envelope : np.ndarray
            Energy envelope (for visualization)
        threshold : float
            Detection threshold used
        """
        # Input validation
        if len(audio) == 0:
            warnings.warn("Audio signal is empty")
            return np.array([]), np.array([]), 0.0
        
        if np.all(audio == 0):
            warnings.warn("Audio signal is silent (all zeros)")
            return np.array([]), np.array([]), 0.0
        
        # Check for clipping
        if np.any(np.abs(audio) >= 0.99):
            warnings.warn("Audio may be clipped - detection accuracy may be reduced")
        
        # Check SNR
        signal_power = np.mean(audio**2)
        if signal_power < 1e-10:
            warnings.warn("Very low signal power - check recording levels")
        
        # Step 1: Bandpass filter to whale click frequencies
        filtered = self.bandpass_filter(audio)
        
        # Step 2: Compute energy envelope
        envelope = self.compute_energy_envelope(filtered)
        
        # Step 3: Find peaks (clicks)
        click_samples, threshold = self.find_click_peaks(envelope)
        
        # Convert sample indices to time in seconds
        click_times = click_samples / self.sample_rate
        
        return click_times, envelope, threshold
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return data with sample rate.
        
        Parameters:
        -----------
        file_path : str
            Path to audio file
            
        Returns:
        --------
        audio : np.ndarray
            Audio data
        sample_rate : int
            Sample rate of the audio
        """
        audio, sample_rate = sf.read(file_path)
        
        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        return audio, sample_rate
    
    def get_detection_stats(self, click_times: np.ndarray, audio_duration: float) -> dict:
        """
        Calculate statistics about the detection results.
        
        Parameters:
        -----------
        click_times : np.ndarray
            Detected click times in seconds
        audio_duration : float
            Total audio duration in seconds
            
        Returns:
        --------
        dict
            Statistics including click count, rate, and intervals
        """
        if len(click_times) == 0:
            return {
                'num_clicks': 0,
                'click_rate': 0.0,
                'mean_interval': 0.0,
                'std_interval': 0.0,
                'duration': audio_duration
            }
        
        # Inter-click intervals
        intervals = np.diff(click_times) if len(click_times) > 1 else np.array([])
        
        stats = {
            'num_clicks': len(click_times),
            'click_rate': len(click_times) / audio_duration,
            'mean_interval': np.mean(intervals) if len(intervals) > 0 else 0.0,
            'std_interval': np.std(intervals) if len(intervals) > 0 else 0.0,
            'duration': audio_duration,
            'first_click': click_times[0],
            'last_click': click_times[-1]
        }
        
        return stats
    
    def _validate_parameters(self) -> None:
        """
        Validate detector parameters and warn about potential issues.
        
        References:
        -----------
        - Watkins & Schevill 1977: click frequency characteristics
        - Madsen et al. 2002: click production mechanics
        - Gero et al. 2016: Caribbean sperm whale acoustics
        """
        if self.actual_lowcut < 1000:
            warnings.warn(
                f"Low frequency cutoff {self.actual_lowcut}Hz is below typical "
                "sperm whale click range. May include ocean noise."
            )
        
        if self.actual_highcut > 25000:
            warnings.warn(
                f"High frequency cutoff {self.actual_highcut}Hz is above typical "
                "sperm whale click range. May include electronic noise."
            )
        
        if self.threshold_multiplier < 2.0:
            warnings.warn(
                f"Threshold multiplier {self.threshold_multiplier} may be too low, "
                "leading to false positive detections."
            )
        
        if self.threshold_multiplier > 5.0:
            warnings.warn(
                f"Threshold multiplier {self.threshold_multiplier} may be too high, "
                "potentially missing weak clicks."
            )
        
        if self.min_click_separation < 0.005:
            warnings.warn(
                f"Minimum click separation {self.min_click_separation*1000:.1f}ms "
                "is below whale click production limits (~5ms)."
            )
    
    def get_parameter_summary(self) -> Dict:
        """
        Get a summary of all detection parameters for transparency.
        
        Returns:
        --------
        Dict
            Complete parameter summary with scientific justification
        """
        return {
            'frequency_filter': self.filter_params,
            'detection_params': {
                'threshold_multiplier': self.threshold_multiplier,
                'min_click_separation_ms': self.min_click_separation * 1000,
                'sample_rate': self.sample_rate
            },
            'scientific_basis': {
                'frequency_range': '2-20kHz based on Gero et al. 2016',
                'threshold_method': '3-sigma detection for 99.7% confidence',
                'separation_limit': 'Madsen 2002: min ICI ~10ms'
            }
        }


def process_audio_file(file_path: str, **detector_kwargs) -> dict:
    """
    Convenience function to process a single audio file.
    
    Parameters:
    -----------
    file_path : str
        Path to audio file
    **detector_kwargs
        Additional arguments for ClickDetector
        
    Returns:
    --------
    dict
        Results including click times, stats, and processing info
    """
    # Load audio
    audio, sample_rate = sf.read(file_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Create detector
    detector = ClickDetector(sample_rate, **detector_kwargs)
    
    # Detect clicks
    click_times, envelope, threshold = detector.detect_clicks(audio)
    
    # Calculate stats
    duration = len(audio) / sample_rate
    stats = detector.get_detection_stats(click_times, duration)
    
    return {
        'file_path': file_path,
        'click_times': click_times,
        'envelope': envelope,
        'threshold': threshold,
        'sample_rate': sample_rate,
        'duration': duration,
        'stats': stats,
        'detector_params': detector.get_parameter_summary()
    }


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # This would be run with actual whale audio files
    print("Click detector module loaded successfully!")
    print("Example usage:")
    print("  detector = ClickDetector(sample_rate=44100)")
    print("  audio, sr = detector.load_audio('whale_recording.wav')")
    print("  clicks, envelope, threshold = detector.detect_clicks(audio)")