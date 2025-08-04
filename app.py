"""
🐋 Sperm Whale Phonetic Alphabet Explorer

Interactive web interface for exploring the newly discovered phonetic alphabet
of sperm whale communication. This app allows researchers and enthusiasts to:

- Upload whale recordings or use sample data
- Detect clicks and group them into codas
- Extract phonetic features (rhythm, tempo, rubato, ornamentation)  
- Explore patterns in the whale's communication system
- Listen to individual codas and patterns

Based on the May 2024 research discovering 143+ distinct click combinations
in sperm whale "language."
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from pathlib import Path
import tempfile
import base64

# Import our whale analysis modules
from src.click_detector import ClickDetector
from src.coda_detector import CodaDetector
from src.feature_extractor import FeatureExtractor


# Page configuration
st.set_page_config(
    page_title="🐋 Whale Phonetic Explorer",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'current_audio' not in st.session_state:
    st.session_state.current_audio = None
if 'current_sample_rate' not in st.session_state:
    st.session_state.current_sample_rate = None
if 'last_params' not in st.session_state:
    st.session_state.last_params = None


def load_sample_files():
    """Load available sample files from the data directory."""
    data_dir = Path("data/raw/watkins")
    if data_dir.exists():
        return list(data_dir.glob("*.wav"))
    return []


def analyze_audio(audio_file_path, progress_bar=None, detector_params=None, coda_params=None):
    """
    Analyze whale audio file and return results.
    
    Parameters:
    -----------
    audio_file_path : str or Path
        Path to audio file
    progress_bar : streamlit progress bar, optional
        Progress bar to update during processing
    detector_params : dict, optional
        Parameters for ClickDetector initialization
    coda_params : dict, optional
        Parameters for CodaDetector initialization
        
    Returns:
    --------
    dict
        Analysis results including clicks, codas, and features
    """
    try:
        # Initialize detectors with user parameters
        if detector_params is None:
            detector_params = {}
        if coda_params is None:
            coda_params = {}
            
        detector = ClickDetector(sample_rate=44100, **detector_params)
        coda_detector = CodaDetector(**coda_params)
        extractor = FeatureExtractor()
        
        if progress_bar:
            progress_bar.progress(10, "Loading audio...")
        
        # Load audio
        audio, sr = detector.load_audio(str(audio_file_path))
        duration = len(audio) / sr
        
        if progress_bar:
            progress_bar.progress(30, "Detecting clicks...")
        
        # Detect clicks
        clicks, envelope, threshold = detector.detect_clicks(audio)
        
        if progress_bar:
            progress_bar.progress(60, "Grouping into codas...")
        
        # Group into codas
        codas = coda_detector.group_clicks_to_codas(clicks)
        
        if progress_bar:
            progress_bar.progress(80, "Extracting phonetic features...")
        
        # Extract features
        if codas:
            features_df = extractor.analyze_coda_collection(codas)
            pattern_stats = extractor.get_pattern_statistics(codas)
        else:
            features_df = pd.DataFrame()
            pattern_stats = {}
        
        if progress_bar:
            progress_bar.progress(100, "Analysis complete!")
        
        # Compile results
        results = {
            'audio': audio,
            'sample_rate': sr,
            'duration': duration,
            'clicks': clicks,
            'envelope': envelope,
            'threshold': threshold,
            'codas': codas,
            'features_df': features_df,
            'pattern_stats': pattern_stats,
            'filename': Path(audio_file_path).name
        }
        
        return results
        
    except Exception as e:
        st.error(f"Error analyzing audio: {str(e)}")
        return None


def plot_audio_analysis(results):
    """Create interactive plots for audio analysis."""
    audio = results['audio']
    sr = results['sample_rate']
    clicks = results['clicks']
    envelope = results['envelope']
    threshold = results['threshold']
    
    # Time axis
    time = np.arange(len(audio)) / sr
    envelope_time = np.arange(len(envelope)) / sr
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Audio Waveform & Click Detection', 'Energy Envelope & Threshold'),
        vertical_spacing=0.1
    )
    
    # Top plot: Waveform with detected clicks
    fig.add_trace(
        go.Scatter(x=time, y=audio, name='Audio', line=dict(color='lightblue', width=1)),
        row=1, col=1
    )
    
    if len(clicks) > 0:
        # Mark detected clicks
        click_amplitudes = [audio[int(click * sr)] if int(click * sr) < len(audio) else 0 
                           for click in clicks]
        fig.add_trace(
            go.Scatter(x=clicks, y=click_amplitudes, mode='markers',
                      name=f'Clicks ({len(clicks)})', 
                      marker=dict(color='red', size=8, symbol='triangle-up')),
            row=1, col=1
        )
    
    # Bottom plot: Energy envelope with threshold
    fig.add_trace(
        go.Scatter(x=envelope_time, y=envelope, name='Energy Envelope', 
                  line=dict(color='green', width=2)),
        row=2, col=1
    )
    
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"Threshold: {threshold:.3f}", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=600,
        title=f"Audio Analysis: {results['filename']}",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Energy", row=2, col=1)
    
    return fig


def plot_phonetic_patterns(features_df):
    """Create visualizations for phonetic patterns."""
    if features_df.empty:
        return None
    
    # Create subplots for different pattern aspects
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Rhythm Patterns', 'Tempo Distribution', 
                       'Rubato Levels', 'Pattern Timeline'),
        specs=[[{"type": "bar"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Rhythm patterns
    rhythm_counts = features_df['rhythm_pattern'].value_counts()
    fig.add_trace(
        go.Bar(x=rhythm_counts.index, y=rhythm_counts.values, 
               name='Rhythm Patterns', marker_color='lightcoral'),
        row=1, col=1
    )
    
    # Tempo distribution
    fig.add_trace(
        go.Histogram(x=features_df['tempo_cps'], name='Tempo (cps)',
                    marker_color='lightblue', nbinsx=10),
        row=1, col=2
    )
    
    # Rubato levels
    rubato_counts = features_df['rubato_level'].value_counts()
    fig.add_trace(
        go.Bar(x=rubato_counts.index, y=rubato_counts.values,
               name='Rubato Levels', marker_color='lightgreen'),
        row=2, col=1
    )
    
    # Pattern timeline
    fig.add_trace(
        go.Scatter(x=features_df['start_time'], y=features_df['tempo_cps'],
                  mode='markers', name='Codas', 
                  marker=dict(size=features_df['num_clicks']*2, 
                             color=features_df['rubato_score'],
                             colorscale='Viridis', showscale=True,
                             colorbar=dict(title="Rubato Score")),
                  text=features_df['phonetic_code'],
                  hovertemplate='<b>%{text}</b><br>Time: %{x:.2f}s<br>Tempo: %{y:.2f} cps<extra></extra>'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(height=800, showlegend=False, 
                     title="Phonetic Pattern Analysis")
    
    # Update axes
    fig.update_xaxes(title_text="Pattern", row=1, col=1)
    fig.update_xaxes(title_text="Clicks per Second", row=1, col=2)
    fig.update_xaxes(title_text="Rubato Level", row=2, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=2)
    
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Tempo (cps)", row=2, col=2)
    
    return fig


def create_audio_player(audio, sample_rate, start_time=0, duration=None):
    """Create an audio player for a specific audio segment."""
    if duration is None:
        audio_segment = audio[int(start_time * sample_rate):]
    else:
        start_sample = int(start_time * sample_rate)
        end_sample = int((start_time + duration) * sample_rate)
        audio_segment = audio[start_sample:end_sample]
    
    # Convert to bytes for playback
    audio_bytes = (audio_segment * 32767).astype(np.int16).tobytes()
    
    return audio_bytes, sample_rate


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🐋 Sperm Whale Phonetic Alphabet Explorer")
    st.markdown("""
    Explore the newly discovered phonetic alphabet of sperm whale communication! 
    This tool analyzes whale click patterns to identify the four key components:
    **Rhythm**, **Tempo**, **Rubato**, and **Ornamentation**.
    """)
    
    # Sidebar
    st.sidebar.header("🎵 Audio Source")
    
    # File source selection
    source_type = st.sidebar.radio(
        "Choose audio source:",
        ["Sample Files", "Upload File"],
        key="source_type"
    )
    
    audio_file = None
    
    if source_type == "Sample Files":
        sample_files = load_sample_files()
        if sample_files:
            selected_file = st.sidebar.selectbox(
                "Select a sample recording:",
                sample_files,
                format_func=lambda x: x.name
            )
            audio_file = selected_file
            
            # Show file info
            if selected_file:
                file_size = selected_file.stat().st_size / (1024 * 1024)
                st.sidebar.info(f"**File:** {selected_file.name}  \n**Size:** {file_size:.1f} MB")
        else:
            st.sidebar.warning("No sample files found. Run `python scripts/download_sample_data.py` first.")
    
    else:  # Upload File
        uploaded_file = st.sidebar.file_uploader(
            "Upload a whale recording (WAV format):",
            type=['wav'],
            help="Upload a WAV file containing sperm whale vocalizations",
            key="audio_uploader"
        )
        
        if uploaded_file:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.read())
                audio_file = tmp_file.name
                
            # Show uploaded file info
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.sidebar.info(f"**File:** {uploaded_file.name}  \n**Size:** {file_size:.1f} MB")
    
    # Analysis controls
    st.sidebar.header("🔧 Analysis Settings")
    
    # Click detection parameters
    st.sidebar.subheader("Click Detection")
    threshold_multiplier = st.sidebar.slider(
        "Detection Sensitivity", 
        min_value=1.0, max_value=5.0, value=3.0, step=0.1,
        help="Higher values = fewer false positives, lower values = catch more clicks",
        key="threshold_multiplier"
    )
    
    # Coda detection parameters  
    st.sidebar.subheader("Coda Grouping")
    max_ici = st.sidebar.slider(
        "Max Inter-Click Interval (s)",
        min_value=0.5, max_value=5.0, value=2.0, step=0.1,
        help="Maximum time between clicks in the same coda",
        key="max_ici"
    )
    
    min_clicks = st.sidebar.slider(
        "Minimum Clicks per Coda",
        min_value=2, max_value=10, value=3, step=1,
        help="Minimum number of clicks to form a valid coda",
        key="min_clicks"
    )
    
    # Create current parameter set for comparison
    current_params = {
        'threshold_multiplier': threshold_multiplier,
        'max_ici': max_ici,
        'min_clicks': min_clicks
    }
    
    # Check if parameters have changed
    params_changed = (st.session_state.last_params != current_params)
    
    # Show parameter status
    if params_changed and st.session_state.last_params is not None:
        st.sidebar.info("⚙️ Parameters changed. Click 'Analyze Audio' to apply new settings.")
    
    # Show current parameter values
    with st.sidebar.expander("📊 Current Parameter Values", expanded=False):
        st.write(f"**Detection Sensitivity:** {threshold_multiplier:.1f}")
        st.write(f"**Max Inter-Click Interval:** {max_ici:.1f}s")
        st.write(f"**Min Clicks per Coda:** {min_clicks}")
        
        if st.session_state.last_params is not None:
            st.write("**Last Analysis Used:**")
            st.write(f"- Sensitivity: {st.session_state.last_params['threshold_multiplier']:.1f}")
            st.write(f"- Max ICI: {st.session_state.last_params['max_ici']:.1f}s")
            st.write(f"- Min Clicks: {st.session_state.last_params['min_clicks']}")
    
    # Analysis button
    if audio_file and st.sidebar.button("🔍 Analyze Audio", type="primary"):
        with st.spinner("Analyzing whale communication patterns..."):
            progress_bar = st.progress(0)
            
            # Prepare detector parameters
            detector_params = {
                'threshold_multiplier': threshold_multiplier
            }
            
            coda_params = {
                'max_ici': max_ici,
                'min_clicks': min_clicks
            }
            
            # Store current parameters
            st.session_state.last_params = current_params.copy()
            
            # Analyze audio with user parameters
            results = analyze_audio(audio_file, progress_bar, detector_params, coda_params)
            
            if results:
                st.session_state.analysis_results = results
                st.session_state.current_audio = results['audio']
                st.session_state.current_sample_rate = results['sample_rate']
                progress_bar.empty()
                st.success(f"✅ Analysis complete! Found {len(results['clicks'])} clicks in {len(results['codas'])} codas.")
                
                # Show parameter confirmation
                st.sidebar.success("✅ New parameters applied successfully!")
                
                # Add analysis info to results for transparency
                results['analysis_params'] = {
                    'detector_params': detector_params,
                    'coda_params': coda_params,
                    'analysis_timestamp': pd.Timestamp.now()
                }
    
    # Display results
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Summary metrics
        st.header("📊 Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Duration", f"{results['duration']:.1f}s")
        with col2:
            st.metric("Total Clicks", len(results['clicks']))
        with col3:
            st.metric("Codas Found", len(results['codas']))
        with col4:
            if not results['features_df'].empty:
                unique_patterns = results['features_df']['phonetic_code'].nunique()
                st.metric("Unique Patterns", unique_patterns)
            else:
                st.metric("Unique Patterns", 0)
        
        # Show analysis parameters used
        if 'analysis_params' in results:
            with st.expander("⚙️ Analysis Parameters Used", expanded=False):
                params = results['analysis_params']
                st.write("**Click Detection:**")
                st.write(f"- Threshold Multiplier: {params['detector_params']['threshold_multiplier']:.1f}")
                st.write("**Coda Grouping:**")
                st.write(f"- Max Inter-Click Interval: {params['coda_params']['max_ici']:.1f}s")
                st.write(f"- Minimum Clicks per Coda: {params['coda_params']['min_clicks']}")
                st.write(f"**Analysis Time:** {params['analysis_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Audio analysis visualization
        st.header("🎵 Audio Analysis")
        
        fig_audio = plot_audio_analysis(results)
        st.plotly_chart(fig_audio, use_container_width=True)
        
        # Phonetic patterns
        if not results['features_df'].empty:
            st.header("🧬 Phonetic Patterns")
            
            fig_patterns = plot_phonetic_patterns(results['features_df'])
            if fig_patterns:
                st.plotly_chart(fig_patterns, use_container_width=True)
            
            # Pattern details table
            st.subheader("📋 Detected Codas")
            
            # Prepare display dataframe
            display_df = results['features_df'].copy()
            display_df = display_df.round(3)
            
            # Select columns for display
            display_columns = [
                'coda_id', 'start_time', 'duration', 'num_clicks',
                'rhythm_pattern', 'tempo_cps', 'tempo_category',
                'rubato_level', 'has_ornamentation', 'phonetic_code'
            ]
            
            display_df = display_df[display_columns]
            
            # Rename columns for better display
            display_df.columns = [
                'ID', 'Start (s)', 'Duration (s)', 'Clicks',
                'Rhythm', 'Tempo (cps)', 'Tempo Cat.',
                'Rubato', 'Ornament.', 'Phonetic Code'
            ]
            
            st.dataframe(display_df, use_container_width=True)
            
            # Pattern statistics
            if results['pattern_stats']:
                st.subheader("📈 Pattern Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Rhythm Patterns:**")
                    rhythm_dist = results['pattern_stats'].get('rhythm_pattern_distribution', {})
                    for pattern, count in rhythm_dist.items():
                        st.write(f"- {pattern}: {count}")
                
                with col2:
                    st.write("**Tempo Distribution:**")
                    tempo_dist = results['pattern_stats'].get('tempo_distribution', {})
                    for tempo, count in tempo_dist.items():
                        st.write(f"- {tempo}: {count}")
                
                # Overall statistics
                stats = results['pattern_stats']
                st.write("**Overall Statistics:**")
                st.write(f"- Average tempo: {stats.get('mean_tempo', 0):.2f} clicks/second")
                st.write(f"- Average rubato: {stats.get('mean_rubato', 0):.3f}")
                st.write(f"- Ornamentation frequency: {stats.get('ornamentation_frequency', 0):.1%}")
        
        else:
            st.warning("⚠️ No codas detected. Try adjusting the analysis parameters in the sidebar:")
            st.write("**Suggestions:**")
            st.write("- **Lower Detection Sensitivity** (1.5-2.5) to catch more clicks")
            st.write("- **Increase Max Inter-Click Interval** (3-4s) for longer codas")
            st.write("- **Reduce Min Clicks per Coda** (2) for shorter patterns")
            
            if 'analysis_params' in results:
                st.write("**Current Settings:**")
                params = results['analysis_params']
                st.write(f"- Sensitivity: {params['detector_params']['threshold_multiplier']:.1f}")
                st.write(f"- Max ICI: {params['coda_params']['max_ici']:.1f}s")
                st.write(f"- Min Clicks: {params['coda_params']['min_clicks']}")
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.markdown("""
    This tool is based on the groundbreaking May 2024 research that discovered
    sperm whales use a phonetic alphabet with 143+ distinct combinations.
    
    **Four Key Components:**
    - **Rhythm**: Click grouping patterns
    - **Tempo**: Overall clicking speed  
    - **Rubato**: Timing variations
    - **Ornamentation**: Extra decorative clicks
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    🐋 Built for whale conservation and scientific research • 
    Data from Watkins Marine Mammal Database • 
    Open source project
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()