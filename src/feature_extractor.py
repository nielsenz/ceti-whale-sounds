"""
Sperm whale phonetic feature extraction.

This module extracts the four key phonetic components identified in the
May 2024 research on sperm whale communication:

1. **Rhythm**: Click grouping patterns (e.g., "1+3", "2+3")
2. **Tempo**: Overall clicking speed (slow/medium/fast)  
3. **Rubato**: Timing variations within patterns
4. **Ornamentation**: Extra decorative clicks

These features combine to create the whale's "phonetic alphabet" with 143+
distinct combinations, similar to how humans use phonemes to create language.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd

try:
    from .coda_detector import Coda
except ImportError:
    from coda_detector import Coda


class TempoCategory(Enum):
    """Tempo categories based on clicks per second."""
    SLOW = "slow"
    MEDIUM = "medium"  
    FAST = "fast"


class RubatoLevel(Enum):
    """Rubato (timing variation) levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass
class PhoneticFeatures:
    """
    Container for the four phonetic components of a whale coda.
    
    Attributes:
    -----------
    rhythm_pattern : str
        Click grouping pattern (e.g., "1+3", "2+3", "5")
    rhythm_groups : List[int]
        Number of clicks in each rhythm group
    tempo_cps : float
        Tempo in clicks per second
    tempo_category : TempoCategory
        Categorical tempo classification
    rubato_score : float
        Timing irregularity measure (0 = perfectly regular)
    rubato_level : RubatoLevel
        Categorical rubato classification
    ornamentation_count : int
        Number of extra decorative clicks
    has_ornamentation : bool
        Whether ornamentation is present
    phonetic_code : str
        Combined code representing all features
    detection_confidence : float
        Overall detection confidence (0-1)
    rhythm_confidence : float
        Confidence in rhythm pattern detection (0-1)
    classification_confidence : float
        Confidence in feature classification (0-1)
    is_echolocation_likely : bool
        Whether this is likely echolocation vs communication
    """
    rhythm_pattern: str
    rhythm_groups: List[int]
    tempo_cps: float
    tempo_category: TempoCategory
    rubato_score: float
    rubato_level: RubatoLevel
    ornamentation_count: int
    has_ornamentation: bool
    phonetic_code: str
    detection_confidence: float = 0.0
    rhythm_confidence: float = 0.0
    classification_confidence: float = 0.0
    is_echolocation_likely: bool = False


@dataclass
class ClickDetection:
    """
    Enhanced click detection with confidence measures.
    
    Attributes:
    -----------
    time : float
        Click time in seconds
    amplitude : float
        Click amplitude (normalized)
    snr : float
        Signal-to-noise ratio
    confidence : float
        Detection confidence (0-1)
    """
    time: float
    amplitude: float
    snr: float = 0.0
    confidence: float = 0.0


class FeatureExtractor:
    """
    Extracts phonetic features from whale codas.
    
    This class implements the feature extraction methods described in the
    sperm whale phonetic alphabet research, providing quantitative measures
    of rhythm, tempo, rubato, and ornamentation.
    
    Parameters:
    -----------
    tempo_slow_threshold : float, default=2.0
        Threshold for slow tempo (clicks per second)
    tempo_fast_threshold : float, default=4.0
        Threshold for fast tempo (clicks per second)
    rubato_moderate_threshold : float, default=0.15
        Threshold for moderate rubato level
    rubato_high_threshold : float, default=0.30
        Threshold for high rubato level
    rhythm_break_factor : float, default=2.0
        Factor for detecting rhythm breaks (ICI > mean + factor * std)
        FIXED: Now uses standard deviation for more robust detection
    ornamentation_base_length : int, default=5
        Expected base length for detecting ornamentation
    """
    
    def __init__(
        self,
        tempo_slow_threshold: float = 2.0,
        tempo_fast_threshold: float = 4.0,
        rubato_moderate_threshold: float = 0.15,
        rubato_high_threshold: float = 0.30,
        rhythm_break_factor: float = 2.0,  # More conservative threshold
        ornamentation_base_length: int = 5
    ):
        self.tempo_slow_threshold = tempo_slow_threshold
        self.tempo_fast_threshold = tempo_fast_threshold
        self.rubato_moderate_threshold = rubato_moderate_threshold
        self.rubato_high_threshold = rubato_high_threshold
        self.rhythm_break_factor = rhythm_break_factor
        self.ornamentation_base_length = ornamentation_base_length
    
    def extract_rhythm_pattern(self, coda: Coda) -> Tuple[str, List[int]]:
        """
        Extract rhythm pattern from coda click timing.
        
        Rhythm patterns are identified by finding groups of clicks separated
        by longer pauses. For example, clicks at [0, 0.2, 0.8, 1.0, 1.2]
        might be grouped as "2+3" (2 clicks, pause, 3 clicks).
        
        FIXED: Corrected off-by-one error in group counting logic.
        
        Parameters:
        -----------
        coda : Coda
            Coda to analyze
            
        Returns:
        --------
        pattern : str
            Rhythm pattern string (e.g., "1+3", "2+3", "5")
        groups : List[int]
            Number of clicks in each group
        """
        if coda.num_clicks <= 2:
            return str(coda.num_clicks), [coda.num_clicks]
        
        # Calculate inter-click intervals
        icis = coda.inter_click_intervals
        mean_ici = np.mean(icis)
        std_ici = np.std(icis)
        
        # More conservative threshold to avoid detecting rubato as rhythm breaks
        rhythm_break_threshold = mean_ici + 2.0 * std_ici
        
        # Find rhythm breaks
        rhythm_breaks = []
        for i, ici in enumerate(icis):
            if ici > rhythm_break_threshold:
                rhythm_breaks.append(i)
        
        # Count clicks properly - FIXED off-by-one error
        groups = []
        current_group_start = 0
        
        for break_idx in rhythm_breaks:
            # Group includes clicks from start to break position (inclusive)
            group_size = break_idx - current_group_start + 1
            groups.append(group_size)
            # Next group starts after the break
            current_group_start = break_idx + 1
        
        # Final group - count remaining clicks
        final_group_size = coda.num_clicks - current_group_start
        if final_group_size > 0:
            groups.append(final_group_size)
        
        # Fallback if no groups detected
        if not groups:
            groups = [coda.num_clicks]
        
        pattern = "+".join(map(str, groups))
        
        return pattern, groups
    
    def extract_tempo(self, coda: Coda) -> Tuple[float, TempoCategory]:
        """
        Extract tempo features from coda.
        
        Tempo is measured as clicks per second for biological interpretation.
        FIXED: Now calculates actual clicks per second, not intervals per second.
        
        Parameters:
        -----------
        coda : Coda
            Coda to analyze
            
        Returns:
        --------
        tempo_cps : float
            Tempo in clicks per second (biological interpretation)
        tempo_category : TempoCategory
            Categorical classification
        """
        if coda.num_clicks <= 1 or coda.duration <= 0:
            return 0.0, TempoCategory.SLOW
        
        # FIXED: Calculate tempo as actual clicks per second
        # For biological interpretation: "5 clicks per second" means 5 clicks in 1 second
        tempo_cps = coda.num_clicks / coda.duration
        
        # Note: For technical analysis, interval rate would be:
        # interval_rate = (coda.num_clicks - 1) / coda.duration
        
        # Categorize tempo
        if tempo_cps < self.tempo_slow_threshold:
            category = TempoCategory.SLOW
        elif tempo_cps < self.tempo_fast_threshold:
            category = TempoCategory.MEDIUM
        else:
            category = TempoCategory.FAST
        
        return tempo_cps, category
    
    def extract_rubato(self, coda: Coda) -> Tuple[float, RubatoLevel]:
        """
        Extract rubato (timing variation) features from coda.
        
        Rubato measures how much the timing deviates from perfect regularity.
        It's calculated as the coefficient of variation of inter-click intervals.
        Higher rubato indicates more "expressive" or irregular timing.
        
        Parameters:
        -----------
        coda : Coda
            Coda to analyze
            
        Returns:
        --------
        rubato_score : float
            Timing irregularity measure (coefficient of variation)
        rubato_level : RubatoLevel
            Categorical classification
        """
        if coda.num_clicks <= 2:
            return 0.0, RubatoLevel.LOW
        
        icis = coda.inter_click_intervals
        
        # Calculate coefficient of variation (std / mean)
        mean_ici = np.mean(icis)
        std_ici = np.std(icis)
        
        if mean_ici > 0:
            rubato_score = std_ici / mean_ici
        else:
            rubato_score = 0.0
        
        # Categorize rubato level
        if rubato_score < self.rubato_moderate_threshold:
            level = RubatoLevel.LOW
        elif rubato_score < self.rubato_high_threshold:
            level = RubatoLevel.MODERATE
        else:
            level = RubatoLevel.HIGH
        
        return rubato_score, level
    
    def extract_ornamentation(self, coda: Coda, rhythm_pattern: str = None) -> Tuple[int, bool]:
        """
        Extract ornamentation features from coda.
        
        Ornamentation refers to extra decorative clicks beyond the basic
        pattern structure. FIXED: Now detects ornamentation relative to
        expected pattern length, not a fixed baseline.
        
        Parameters:
        -----------
        coda : Coda
            Coda to analyze
        rhythm_pattern : str, optional
            Rhythm pattern for context-aware detection
            
        Returns:
        --------
        ornamentation_count : int
            Number of extra clicks beyond expected pattern
        has_ornamentation : bool
            Whether ornamentation is present
        """
        # Define expected lengths for known whale patterns
        expected_lengths = {
            '1+3': 4,
            '2+3': 5,
            '1+4': 5,
            '3+1': 4,
            '1+1+3': 5,
            '2+1+3': 6,
            '5R': 5,   # Regular 5-click pattern
            '4R': 4,   # Regular 4-click pattern
            '6R': 6,   # Regular 6-click pattern
            '7R': 7,   # Regular 7-click pattern
            '8R': 8,   # Regular 8-click pattern
            '3': 3,    # Simple 3-click pattern
            '4': 4,    # Simple 4-click pattern
            '5': 5,    # Simple 5-click pattern
            '6': 6,    # Simple 6-click pattern
            '7': 7,    # Simple 7-click pattern
        }
        
        # If rhythm pattern is provided, use it for context
        if rhythm_pattern and rhythm_pattern in expected_lengths:
            expected = expected_lengths[rhythm_pattern]
        elif rhythm_pattern:
            # For unknown patterns, sum the groups
            try:
                groups = rhythm_pattern.split('+')
                expected = sum(int(g.replace('R', '')) for g in groups if g.replace('R', '').isdigit())
            except (ValueError, AttributeError):
                expected = self.ornamentation_base_length
        else:
            # Fallback to original approach
            expected = self.ornamentation_base_length
        
        extra_clicks = max(0, coda.num_clicks - expected)
        has_ornamentation = extra_clicks > 0
        
        return extra_clicks, has_ornamentation
    
    def is_likely_echolocation(self, coda: Coda) -> bool:
        """
        Check if this is likely echolocation rather than communication.
        
        Echolocation characteristics:
        - Very regular inter-click intervals (low coefficient of variation)
        - High click rate (>8 clicks/second)
        - Long sequences (>10 clicks)
        - No rhythm pattern variation
        
        Parameters:
        -----------
        coda : Coda
            Coda to analyze
            
        Returns:
        --------
        bool
            True if likely echolocation, False if likely communication
        """
        if coda.num_clicks < 10:
            return False
        
        if len(coda.inter_click_intervals) == 0:
            return False
            
        # Calculate regularity (coefficient of variation)
        icis = coda.inter_click_intervals
        mean_ici = np.mean(icis)
        std_ici = np.std(icis)
        cv_ici = std_ici / mean_ici if mean_ici > 0 else 0
        
        # Calculate click rate
        click_rate = coda.num_clicks / coda.duration if coda.duration > 0 else 0
        
        # Echolocation criteria
        is_regular = cv_ici < 0.1  # Very regular timing
        is_fast = click_rate > 8   # Fast clicking
        is_long = coda.num_clicks > 20  # Extended sequence
        
        # Need at least two criteria for echolocation classification
        criteria_met = sum([is_regular, is_fast, is_long])
        
        return criteria_met >= 2
    
    def calculate_confidence_scores(self, coda: Coda, rhythm_pattern: str, rhythm_groups: List[int]) -> Tuple[float, float, float]:
        """
        Calculate confidence scores for detection and classification.
        
        Parameters:
        -----------
        coda : Coda
            Coda to analyze
        rhythm_pattern : str
            Detected rhythm pattern
        rhythm_groups : List[int]
            Rhythm group sizes
            
        Returns:
        --------
        detection_confidence : float
            Overall detection confidence (0-1)
        rhythm_confidence : float
            Rhythm pattern confidence (0-1)
        classification_confidence : float
            Feature classification confidence (0-1)
        """
        # Detection confidence based on click count and ICI consistency
        detection_conf = min(1.0, coda.num_clicks / 10.0)  # Higher confidence with more clicks
        
        # Rhythm confidence based on ICI pattern consistency
        if len(coda.inter_click_intervals) > 1:
            icis = coda.inter_click_intervals
            cv = np.std(icis) / np.mean(icis) if np.mean(icis) > 0 else 1.0
            # Lower CV = higher confidence in rhythm detection
            rhythm_conf = max(0.1, 1.0 - cv)
        else:
            rhythm_conf = 0.5
        
        # Classification confidence based on pattern recognition
        known_patterns = {'1+3', '2+3', '1+4', '3+1', '5', '4', '6', '7'}
        if rhythm_pattern in known_patterns:
            class_conf = 0.9
        elif len(rhythm_groups) <= 3:  # Simple patterns
            class_conf = 0.7
        else:  # Complex patterns
            class_conf = 0.5
        
        return detection_conf, rhythm_conf, class_conf
    
    def create_phonetic_code(
        self, 
        rhythm_pattern: str,
        tempo_category: TempoCategory,
        rubato_level: RubatoLevel,
        has_ornamentation: bool
    ) -> str:
        """
        Create a combined phonetic code representing all features.
        
        The phonetic code combines all four features into a single string
        that uniquely identifies the communication pattern.
        
        Parameters:
        -----------
        rhythm_pattern : str
            Rhythm pattern (e.g., "1+3")
        tempo_category : TempoCategory
            Tempo classification
        rubato_level : RubatoLevel
            Rubato classification
        has_ornamentation : bool
            Whether ornamentation is present
            
        Returns:
        --------
        str
            Combined phonetic code
        """
        code_parts = [
            rhythm_pattern,
            tempo_category.value,
            rubato_level.value
        ]
        
        if has_ornamentation:
            code_parts.append("orn")
        
        return "_".join(code_parts)
    
    def extract_all_features(self, coda: Coda) -> PhoneticFeatures:
        """
        Extract all phonetic features from a coda.
        
        This is the main function that combines all feature extraction
        methods to provide a complete phonetic analysis.
        
        Parameters:
        -----------
        coda : Coda
            Coda to analyze
            
        Returns:
        --------
        PhoneticFeatures
            Complete feature analysis
        """
        # Extract each feature component
        rhythm_pattern, rhythm_groups = self.extract_rhythm_pattern(coda)
        tempo_cps, tempo_category = self.extract_tempo(coda)
        rubato_score, rubato_level = self.extract_rubato(coda)
        # FIXED: Pass rhythm pattern for context-aware ornamentation detection
        ornamentation_count, has_ornamentation = self.extract_ornamentation(coda, rhythm_pattern)
        
        # Calculate confidence scores
        detection_conf, rhythm_conf, class_conf = self.calculate_confidence_scores(
            coda, rhythm_pattern, rhythm_groups
        )
        
        # Check if likely echolocation
        is_echolocation = self.is_likely_echolocation(coda)
        
        # Create combined phonetic code
        phonetic_code = self.create_phonetic_code(
            rhythm_pattern, tempo_category, rubato_level, has_ornamentation
        )
        
        return PhoneticFeatures(
            rhythm_pattern=rhythm_pattern,
            rhythm_groups=rhythm_groups,
            tempo_cps=tempo_cps,
            tempo_category=tempo_category,
            rubato_score=rubato_score,
            rubato_level=rubato_level,
            ornamentation_count=ornamentation_count,
            has_ornamentation=has_ornamentation,
            phonetic_code=phonetic_code,
            detection_confidence=detection_conf,
            rhythm_confidence=rhythm_conf,
            classification_confidence=class_conf,
            is_echolocation_likely=is_echolocation
        )
    
    def analyze_coda_collection(self, codas: List[Coda]) -> pd.DataFrame:
        """
        Analyze a collection of codas and return feature summary.
        
        Parameters:
        -----------
        codas : List[Coda]
            List of codas to analyze
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with features for each coda
        """
        if not codas:
            return pd.DataFrame()
        
        # Extract features for all codas
        features_list = []
        for i, coda in enumerate(codas):
            features = self.extract_all_features(coda)
            
            # Convert to dictionary for DataFrame
            feature_dict = {
                'coda_id': i,
                'start_time': coda.start_time,
                'duration': coda.duration,
                'num_clicks': coda.num_clicks,
                'rhythm_pattern': features.rhythm_pattern,
                'tempo_cps': features.tempo_cps,
                'tempo_category': features.tempo_category.value,
                'rubato_score': features.rubato_score,
                'rubato_level': features.rubato_level.value,
                'ornamentation_count': features.ornamentation_count,
                'has_ornamentation': features.has_ornamentation,
                'phonetic_code': features.phonetic_code,
                'detection_confidence': features.detection_confidence,
                'rhythm_confidence': features.rhythm_confidence,
                'classification_confidence': features.classification_confidence,
                'is_echolocation_likely': features.is_echolocation_likely
            }
            
            features_list.append(feature_dict)
        
        return pd.DataFrame(features_list)
    
    def get_pattern_statistics(self, codas: List[Coda]) -> Dict:
        """
        Generate statistics about phonetic patterns in a collection of codas.
        
        Parameters:
        -----------
        codas : List[Coda]
            List of codas to analyze
            
        Returns:
        --------
        Dict
            Pattern statistics and distributions
        """
        if not codas:
            return {}
        
        df = self.analyze_coda_collection(codas)
        
        stats = {
            'total_codas': len(codas),
            'unique_phonetic_codes': df['phonetic_code'].nunique(),
            'phonetic_code_distribution': df['phonetic_code'].value_counts().to_dict(),
            'rhythm_pattern_distribution': df['rhythm_pattern'].value_counts().to_dict(),
            'tempo_distribution': df['tempo_category'].value_counts().to_dict(),
            'rubato_distribution': df['rubato_level'].value_counts().to_dict(),
            'ornamentation_frequency': df['has_ornamentation'].sum() / len(df),
            'mean_tempo': df['tempo_cps'].mean(),
            'mean_rubato': df['rubato_score'].mean(),
            'mean_duration': df['duration'].mean(),
            'mean_clicks_per_coda': df['num_clicks'].mean()
        }
        
        return stats
    
    def find_similar_patterns(
        self, 
        target_coda: Coda, 
        coda_collection: List[Coda],
        similarity_threshold: float = 0.8
    ) -> List[Tuple[Coda, float]]:
        """
        Find codas with similar phonetic patterns to a target coda.
        
        Parameters:
        -----------
        target_coda : Coda
            Reference coda to match against
        coda_collection : List[Coda]
            Collection of codas to search
        similarity_threshold : float
            Minimum similarity score (0-1)
            
        Returns:
        --------
        List[Tuple[Coda, float]]
            List of (coda, similarity_score) pairs
        """
        target_features = self.extract_all_features(target_coda)
        similar_codas = []
        
        for coda in coda_collection:
            if coda is target_coda:
                continue
                
            features = self.extract_all_features(coda)
            
            # Calculate similarity based on feature matching
            similarity_score = self._calculate_similarity(target_features, features)
            
            if similarity_score >= similarity_threshold:
                similar_codas.append((coda, similarity_score))
        
        # Sort by similarity score (descending)
        similar_codas.sort(key=lambda x: x[1], reverse=True)
        
        return similar_codas
    
    def _calculate_similarity(
        self, 
        features1: PhoneticFeatures, 
        features2: PhoneticFeatures
    ) -> float:
        """
        Calculate similarity between two sets of phonetic features.
        
        Parameters:
        -----------
        features1, features2 : PhoneticFeatures
            Feature sets to compare
            
        Returns:
        --------
        float
            Similarity score (0-1)
        """
        # Exact matches get full points
        similarity_score = 0.0
        
        # Rhythm pattern (40% weight)
        if features1.rhythm_pattern == features2.rhythm_pattern:
            similarity_score += 0.4
        
        # Tempo category (25% weight)
        if features1.tempo_category == features2.tempo_category:
            similarity_score += 0.25
        
        # Rubato level (25% weight) 
        if features1.rubato_level == features2.rubato_level:
            similarity_score += 0.25
        
        # Ornamentation (10% weight)
        if features1.has_ornamentation == features2.has_ornamentation:
            similarity_score += 0.1
        
        return similarity_score


def extract_features_from_clicks(
    click_times: np.ndarray,
    **kwargs
) -> Tuple[List[PhoneticFeatures], pd.DataFrame]:
    """
    Convenience function to extract features directly from click times.
    
    Parameters:
    -----------
    click_times : np.ndarray
        Array of click times in seconds
    **kwargs
        Additional arguments for CodaDetector and FeatureExtractor
        
    Returns:
    --------
    features_list : List[PhoneticFeatures]
        Phonetic features for each detected coda
    features_df : pd.DataFrame
        DataFrame summary of all features
    """
    try:
        from .coda_detector import CodaDetector
    except ImportError:
        from coda_detector import CodaDetector
    
    # Extract detector and extractor kwargs
    detector_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['max_ici', 'min_clicks', 'max_coda_duration']}
    extractor_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in detector_kwargs}
    
    # Detect codas
    detector = CodaDetector(**detector_kwargs)
    codas = detector.group_clicks_to_codas(click_times)
    
    # Extract features
    extractor = FeatureExtractor(**extractor_kwargs)
    features_list = [extractor.extract_all_features(coda) for coda in codas]
    features_df = extractor.analyze_coda_collection(codas)
    
    return features_list, features_df


if __name__ == "__main__":
    # Example usage
    print("Feature extractor module loaded successfully!")
    print("Example usage:")
    print("  extractor = FeatureExtractor()")
    print("  features = extractor.extract_all_features(coda)")
    print("  df = extractor.analyze_coda_collection(codas)")
    
    # Demo with synthetic data
    try:
        from .coda_detector import Coda
    except ImportError:
        from coda_detector import Coda
    
    print("\nDemo with synthetic coda:")
    synthetic_clicks = np.array([1.0, 1.2, 1.4, 2.0, 2.2, 2.4, 2.6])
    coda = Coda(synthetic_clicks)
    
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(coda)
    
    print(f"Input clicks: {synthetic_clicks}")
    print(f"Phonetic code: {features.phonetic_code}")
    print(f"Rhythm: {features.rhythm_pattern}")
    print(f"Tempo: {features.tempo_cps:.2f} cps ({features.tempo_category.value})")
    print(f"Rubato: {features.rubato_score:.3f} ({features.rubato_level.value})")
    print(f"Ornamentation: {features.ornamentation_count} extra clicks")