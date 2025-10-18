import os
import tempfile

import numpy as np
import soundfile as sf

from app import analyze_audio


def _write_wav(temp_dir, sr, seconds=0.25, freq=1000.0):
    n = int(sr * seconds)
    t = np.arange(n) / sr
    # quiet sine to avoid clipping; add a small pulse to simulate a click
    audio = 0.01 * np.sin(2 * np.pi * freq * t)
    if n > 10:
        audio[5:8] += 0.5  # simple transient
    path = os.path.join(temp_dir, f"test_{sr}.wav")
    sf.write(path, audio, sr)
    return path


def test_analyze_audio_uses_actual_sample_rate_low_nyquist():
    # Choose a rate where requested highcut (20000 Hz) exceeds Nyquist
    # so detector should clamp to 0.95 * Nyquist
    with tempfile.TemporaryDirectory() as tmp:
        sr = 32000  # Nyquist=16000 -> actual_highcut should be ~15200
        wav_path = _write_wav(tmp, sr)

        results = analyze_audio(wav_path)
        assert results is not None
        assert results["sample_rate"] == sr

        det = results.get("detector_summary", {})
        assert det and det.get("detection_params", {}).get("sample_rate") == sr

        actual_highcut = det.get("frequency_filter", {}).get("actual", {}).get("highcut")
        assert actual_highcut is not None
        # expected clamp: 0.95 * (sr/2) = 15200
        expected = 0.95 * (sr / 2)
        assert abs(actual_highcut - expected) < 5.0


def test_analyze_audio_uses_actual_sample_rate_high_nyquist():
    # Choose a rate where requested highcut (20000 Hz) is below Nyquist
    with tempfile.TemporaryDirectory() as tmp:
        sr = 48000  # Nyquist=24000 -> actual_highcut should remain 20000
        wav_path = _write_wav(tmp, sr)

        results = analyze_audio(wav_path)
        assert results is not None
        assert results["sample_rate"] == sr

        det = results.get("detector_summary", {})
        assert det and det.get("detection_params", {}).get("sample_rate") == sr

        actual_highcut = det.get("frequency_filter", {}).get("actual", {}).get("highcut")
        assert actual_highcut is not None
        assert abs(actual_highcut - 20000.0) < 1e-6

