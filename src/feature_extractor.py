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

from .coda_detector import Coda


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
    rhythm_break_factor : float, default=1.5
        Factor for detecting rhythm breaks (ICI > mean * factor)
    ornamentation_base_length : int, default=5
        Expected base length for detecting ornamentation
    """
    
    def __init__(
        self,
        tempo_slow_threshold: float = 2.0,
        tempo_fast_threshold: float = 4.0,
        rubato_moderate_threshold: float = 0.15,
        rubato_high_threshold: float = 0.30,
        rhythm_break_factor: float = 1.5,
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
        if coda.num_clicks <= 1:
            return "1", [1]
        
        if coda.num_clicks == 2:
            return "2", [2]
        
        # Calculate inter-click intervals
        icis = coda.inter_click_intervals
        
        # Find rhythm breaks based on longer intervals
        mean_ici = np.mean(icis)
        rhythm_breaks = []
        
        for i, ici in enumerate(icis):
            if ici > self.rhythm_break_factor * mean_ici:
                rhythm_breaks.append(i)
        
        # Group clicks based on breaks
        groups = []
        start_idx = 0
        
        for break_idx in rhythm_breaks:
            group_size = break_idx - start_idx + 1
            groups.append(group_size)
            start_idx = break_idx + 1
        
        # Add final group
        final_group_size = len(icis) - start_idx + 1
        groups.append(final_group_size)
        
        # Create pattern string
        if not groups:
            groups = [coda.num_clicks]
        
        pattern = "+".join(map(str, groups))
        
        return pattern, groups
    
    def extract_tempo(self, coda: Coda) -> Tuple[float, TempoCategory]:
        """
        Extract tempo features from coda.
        
        Tempo is measured as clicks per second, calculated as the number of
        inter-click intervals divided by the total coda duration.
        
        Parameters:
        -----------
        coda : Coda
            Coda to analyze
            
        Returns:
        --------
        tempo_cps : float
            Tempo in clicks per second
        tempo_category : TempoCategory
            Categorical classification
        """
        if coda.num_clicks <= 1 or coda.duration <= 0:
            return 0.0, TempoCategory.SLOW
        
        # Calculate tempo as (n_clicks - 1) / duration
        # This gives us the rate of click intervals per second
        tempo_cps = (coda.num_clicks - 1) / coda.duration
        
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
    
    def extract_ornamentation(self, coda: Coda) -> Tuple[int, bool]:
        """
        Extract ornamentation features from coda.
        
        Ornamentation refers to extra decorative clicks beyond the basic
        pattern structure. This is detected by comparing the coda length
        to expected typical lengths for the rhythm pattern.
        
        Parameters:
        -----------
        coda : Coda
            Coda to analyze
            
        Returns:
        --------
        ornamentation_count : int
            Number of extra clicks beyond base pattern
        has_ornamentation : bool
            Whether ornamentation is present
        """
        # Simple approach: consider codas longer than base length as ornamented
        extra_clicks = max(0, coda.num_clicks - self.ornamentation_base_length)
        has_ornamentation = extra_clicks > 0
        
        return extra_clicks, has_ornamentation
    
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
        ornamentation_count, has_ornamentation = self.extract_ornamentation(coda)
        
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
            phonetic_code=phonetic_code
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
                'phonetic_code': features.phonetic_code
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
    from .coda_detector import CodaDetector
    
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
    from .coda_detector import Coda
    
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