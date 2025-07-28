"""
Sperm whale coda detection and analysis.

This module groups individual whale clicks into codas (communication units) and
analyzes their structure. A coda is a stereotyped pattern of clicks that whales
use for communication, distinct from echolocation clicks.

Key concepts:
- Coda: A sequence of clicks forming a communication pattern
- Inter-Click Interval (ICI): Time between consecutive clicks
- Coda boundary: Determined by longer pauses between click groups
- Minimum coda size: Usually 3+ clicks to be considered meaningful
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Coda:
    """
    Represents a single whale coda with its clicks and derived properties.
    
    Attributes:
    -----------
    click_times : np.ndarray
        Times of clicks in this coda (seconds)
    start_time : float
        Time of first click (seconds)
    end_time : float
        Time of last click (seconds)
    duration : float
        Total coda duration (seconds)
    num_clicks : int
        Number of clicks in the coda
    inter_click_intervals : np.ndarray
        Time intervals between consecutive clicks
    mean_ici : float
        Average inter-click interval
    """
    click_times: np.ndarray
    
    def __post_init__(self):
        """Calculate derived properties after initialization."""
        self.start_time = self.click_times[0]
        self.end_time = self.click_times[-1]
        self.duration = self.end_time - self.start_time
        self.num_clicks = len(self.click_times)
        
        if self.num_clicks > 1:
            self.inter_click_intervals = np.diff(self.click_times)
            self.mean_ici = np.mean(self.inter_click_intervals)
        else:
            self.inter_click_intervals = np.array([])
            self.mean_ici = 0.0


class CodaDetector:
    """
    Groups whale clicks into codas based on temporal patterns.
    
    The detector works by analyzing inter-click intervals (ICIs) to identify
    natural breaks between communication units. Clicks separated by short
    intervals (< max_ici) belong to the same coda, while longer gaps indicate
    coda boundaries.
    
    Parameters:
    -----------
    max_ici : float, default=2.0
        Maximum inter-click interval within a coda (seconds)
    min_clicks : int, default=3
        Minimum number of clicks to form a valid coda
    max_coda_duration : float, default=10.0
        Maximum duration of a single coda (seconds)
    """
    
    def __init__(
        self,
        max_ici: float = 2.0,
        min_clicks: int = 3,
        max_coda_duration: float = 10.0
    ):
        self.max_ici = max_ici
        self.min_clicks = min_clicks
        self.max_coda_duration = max_coda_duration
    
    def group_clicks_to_codas(self, click_times: np.ndarray) -> List[Coda]:
        """
        Group clicks into codas based on inter-click intervals.
        
        The algorithm:
        1. Start with the first click
        2. Add subsequent clicks if they're within max_ici
        3. When a gap > max_ici is found, end current coda
        4. Start new coda with the next click
        5. Filter out codas with < min_clicks
        
        Parameters:
        -----------
        click_times : np.ndarray
            Array of click times in seconds
            
        Returns:
        --------
        List[Coda]
            List of detected codas
        """
        if len(click_times) < self.min_clicks:
            return []
        
        codas = []
        current_coda_clicks = [click_times[0]]
        
        for i in range(1, len(click_times)):
            time_gap = click_times[i] - click_times[i-1]
            current_duration = click_times[i] - current_coda_clicks[0]
            
            # Check if click belongs to current coda
            if (time_gap <= self.max_ici and 
                current_duration <= self.max_coda_duration):
                current_coda_clicks.append(click_times[i])
            else:
                # End current coda if it has enough clicks
                if len(current_coda_clicks) >= self.min_clicks:
                    codas.append(Coda(np.array(current_coda_clicks)))
                
                # Start new coda
                current_coda_clicks = [click_times[i]]
        
        # Don't forget the last coda
        if len(current_coda_clicks) >= self.min_clicks:
            codas.append(Coda(np.array(current_coda_clicks)))
        
        return codas
    
    def analyze_coda_structure(self, coda: Coda) -> Dict:
        """
        Analyze the internal structure of a coda.
        
        This function extracts features that will be used for phonetic analysis:
        - Basic timing statistics
        - Rhythm patterns (groups of clicks)
        - Tempo (clicks per second)
        - Timing regularity
        
        Parameters:
        -----------
        coda : Coda
            Coda to analyze
            
        Returns:
        --------
        Dict
            Dictionary of structural features
        """
        analysis = {
            'num_clicks': coda.num_clicks,
            'duration': coda.duration,
            'start_time': coda.start_time,
            'end_time': coda.end_time,
            'mean_ici': coda.mean_ici,
        }
        
        if coda.num_clicks > 1:
            icis = coda.inter_click_intervals
            
            # Timing statistics
            analysis.update({
                'std_ici': np.std(icis),
                'min_ici': np.min(icis),
                'max_ici': np.max(icis),
                'ici_range': np.max(icis) - np.min(icis),
                'cv_ici': np.std(icis) / np.mean(icis) if np.mean(icis) > 0 else 0,
            })
            
            # Tempo (clicks per second, excluding start/end gaps)
            if coda.duration > 0:
                # Use (n_clicks - 1) / duration for internal tempo
                analysis['tempo_cps'] = (coda.num_clicks - 1) / coda.duration
            else:
                analysis['tempo_cps'] = 0.0
            
            # Basic rhythm detection (find groups separated by longer pauses)
            rhythm_groups = self._detect_rhythm_groups(icis)
            analysis['rhythm_pattern'] = rhythm_groups
            analysis['rhythm_string'] = '+'.join(map(str, rhythm_groups))
            
        else:
            # Single click "coda" (unusual but possible)
            analysis.update({
                'std_ici': 0.0,
                'min_ici': 0.0,
                'max_ici': 0.0,
                'ici_range': 0.0,
                'cv_ici': 0.0,
                'tempo_cps': 0.0,
                'rhythm_pattern': [1],
                'rhythm_string': '1'
            })
        
        return analysis
    
    def _detect_rhythm_groups(self, icis: np.ndarray) -> List[int]:
        """
        Detect rhythm groups within a coda based on ICI patterns.
        
        This is a simple version that looks for ICIs significantly longer
        than the average, which may indicate rhythm breaks.
        
        Parameters:
        -----------
        icis : np.ndarray
            Inter-click intervals
            
        Returns:
        --------
        List[int]
            Size of each rhythm group
        """
        if len(icis) == 0:
            return [1]
        
        if len(icis) == 1:
            return [2]  # Two clicks = one group
        
        # Find rhythm breaks (ICIs that are significantly longer)
        mean_ici = np.mean(icis)
        std_ici = np.std(icis)
        
        # An ICI is a rhythm break if it's > mean + 0.5*std
        # This is a conservative threshold to avoid over-segmentation
        rhythm_break_threshold = mean_ici + 0.5 * std_ici
        
        rhythm_breaks = []
        for i, ici in enumerate(icis):
            if ici > rhythm_break_threshold:
                rhythm_breaks.append(i)
        
        # Count clicks in each group
        groups = []
        start_idx = 0
        
        for break_idx in rhythm_breaks:
            group_size = break_idx - start_idx + 1
            groups.append(group_size)
            start_idx = break_idx + 1
        
        # Add the final group
        final_group_size = len(icis) - start_idx + 1
        groups.append(final_group_size)
        
        return groups if groups else [len(icis) + 1]
    
    def get_detection_summary(self, codas: List[Coda]) -> Dict:
        """
        Generate summary statistics for a set of detected codas.
        
        Parameters:
        -----------
        codas : List[Coda]
            List of detected codas
            
        Returns:
        --------
        Dict
            Summary statistics
        """
        if not codas:
            return {
                'num_codas': 0,
                'total_clicks': 0,
                'mean_coda_length': 0.0,
                'mean_coda_duration': 0.0,
                'mean_tempo': 0.0
            }
        
        # Extract basic metrics
        coda_lengths = [coda.num_clicks for coda in codas]
        coda_durations = [coda.duration for coda in codas]
        total_clicks = sum(coda_lengths)
        
        # Calculate tempos (exclude zero-duration codas)
        tempos = []
        for coda in codas:
            if coda.duration > 0 and coda.num_clicks > 1:
                tempo = (coda.num_clicks - 1) / coda.duration
                tempos.append(tempo)
        
        summary = {
            'num_codas': len(codas),
            'total_clicks': total_clicks,
            'mean_coda_length': np.mean(coda_lengths),
            'std_coda_length': np.std(coda_lengths),
            'min_coda_length': np.min(coda_lengths),
            'max_coda_length': np.max(coda_lengths),
            'mean_coda_duration': np.mean(coda_durations),
            'std_coda_duration': np.std(coda_durations),
            'mean_tempo': np.mean(tempos) if tempos else 0.0,
            'std_tempo': np.std(tempos) if tempos else 0.0,
        }
        
        # Rhythm pattern distribution
        rhythm_patterns = []
        for coda in codas:
            analysis = self.analyze_coda_structure(coda)
            rhythm_patterns.append(analysis['rhythm_string'])
        
        # Count unique patterns
        unique_patterns = {}
        for pattern in rhythm_patterns:
            unique_patterns[pattern] = unique_patterns.get(pattern, 0) + 1
        
        summary['rhythm_patterns'] = unique_patterns
        summary['num_unique_patterns'] = len(unique_patterns)
        
        return summary
    
    def filter_codas(
        self, 
        codas: List[Coda], 
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        min_tempo: Optional[float] = None,
        max_tempo: Optional[float] = None
    ) -> List[Coda]:
        """
        Filter codas based on various criteria.
        
        Parameters:
        -----------
        codas : List[Coda]
            Input codas to filter
        min_duration : float, optional
            Minimum coda duration (seconds)
        max_duration : float, optional
            Maximum coda duration (seconds)
        min_tempo : float, optional
            Minimum tempo (clicks per second)
        max_tempo : float, optional
            Maximum tempo (clicks per second)
            
        Returns:
        --------
        List[Coda]
            Filtered codas
        """
        filtered = []
        
        for coda in codas:
            # Duration filters
            if min_duration is not None and coda.duration < min_duration:
                continue
            if max_duration is not None and coda.duration > max_duration:
                continue
            
            # Tempo filters
            if coda.duration > 0 and coda.num_clicks > 1:
                tempo = (coda.num_clicks - 1) / coda.duration
                if min_tempo is not None and tempo < min_tempo:
                    continue
                if max_tempo is not None and tempo > max_tempo:
                    continue
            
            filtered.append(coda)
        
        return filtered


def process_clicks_to_codas(
    click_times: np.ndarray, 
    **detector_kwargs
) -> Tuple[List[Coda], Dict]:
    """
    Convenience function to process clicks into codas with analysis.
    
    Parameters:
    -----------
    click_times : np.ndarray
        Array of click times in seconds
    **detector_kwargs
        Arguments for CodaDetector
        
    Returns:
    --------
    codas : List[Coda]
        Detected codas
    summary : Dict
        Summary statistics
    """
    detector = CodaDetector(**detector_kwargs)
    codas = detector.group_clicks_to_codas(click_times)
    summary = detector.get_detection_summary(codas)
    
    # Add detailed analysis for each coda
    coda_analyses = []
    for coda in codas:
        analysis = detector.analyze_coda_structure(coda)
        coda_analyses.append(analysis)
    
    summary['coda_analyses'] = coda_analyses
    
    return codas, summary


if __name__ == "__main__":
    # Example usage
    print("Coda detector module loaded successfully!")
    print("Example usage:")
    print("  detector = CodaDetector(max_ici=2.0, min_clicks=3)")
    print("  codas = detector.group_clicks_to_codas(click_times)")
    print("  summary = detector.get_detection_summary(codas)")
    
    # Demo with synthetic data
    print("\nDemo with synthetic click times:")
    synthetic_clicks = np.array([1.0, 1.2, 1.4, 4.0, 4.3, 4.5, 4.8, 8.0, 8.1, 8.3])
    detector = CodaDetector()
    codas = detector.group_clicks_to_codas(synthetic_clicks)
    
    print(f"Input clicks: {synthetic_clicks}")
    print(f"Detected {len(codas)} codas:")
    for i, coda in enumerate(codas):
        analysis = detector.analyze_coda_structure(coda)
        print(f"  Coda {i+1}: {coda.num_clicks} clicks, pattern '{analysis['rhythm_string']}'")