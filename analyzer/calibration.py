"""
Calibration module for EchoNest-compatible audio analysis output.

This module loads and applies calibration data to transform raw analysis
features into values that match the EchoNest/Spotify audio analysis format.

Supports the newer calibration.json format with:
- Timbre affine transform (scale a + bias b)
- Loudness calibration
- Confidence remapping curves
- Pitch weights and normalization
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np


@dataclass
class TimbreCalibration:
    """Affine transformation for timbre coefficients."""
    a: List[float] = field(default_factory=lambda: [1.0] * 12)  # Scale
    b: List[float] = field(default_factory=lambda: [0.0] * 12)  # Bias


@dataclass
class LoudnessCalibration:
    """Affine transforms for loudness values."""
    start_a: float = 1.0
    start_b: float = 0.0
    max_a: float = 1.0
    max_b: float = 0.0


@dataclass
class ConfidenceCalibration:
    """Piecewise linear mapping for confidence values."""
    source: List[float] = field(default_factory=lambda: [0.0, 1.0])
    target: List[float] = field(default_factory=lambda: [0.0, 1.0])


@dataclass
class PitchCalibration:
    """Pitch vector calibration."""
    power: float = 1.0  # Power transform exponent
    weights: List[float] = field(default_factory=lambda: [1.0] * 12)
    normalize: str = "max"  # "max" or "l1" or "l2"


@dataclass
class SegmentationConfig:
    """Segmentation parameters."""
    min_segment_duration: float = 0.25
    novelty_smoothing: int = 8
    peak_threshold: float = 0.3
    peak_prominence: float = 0.2
    max_segments_per_second: float = 2.5
    beat_snap_tolerance: float = 0.12


@dataclass
class FeatureConfig:
    """Feature extraction parameters."""
    sample_rate: int = 44100
    frame_size: int = 2048
    hop_size: int = 512


@dataclass
class CalibrationConfig:
    """Complete calibration configuration."""
    timbre: TimbreCalibration = field(default_factory=TimbreCalibration)
    loudness: LoudnessCalibration = field(default_factory=LoudnessCalibration)
    confidence: ConfidenceCalibration = field(default_factory=ConfidenceCalibration)
    pitch: PitchCalibration = field(default_factory=PitchCalibration)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    tatums_per_beat: int = 2
    time_signature: int = 4


class AnalysisCalibrator:
    """
    Applies calibration transforms to audio analysis output.
    
    Usage:
        calibrator = AnalysisCalibrator("calibration.json")
        calibrated_timbre = calibrator.calibrate_timbre(raw_mfcc)
        calibrated_pitches = calibrator.calibrate_pitches(raw_chroma)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize calibrator.
        
        Args:
            config_path: Path to calibration.json file.
                         If None, uses default (no-op) calibration.
        """
        self.config = CalibrationConfig()
        
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
    
    def _load_config(self, path: str) -> None:
        """Load calibration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        # Timbre
        if "timbre" in data:
            t = data["timbre"]
            self.config.timbre = TimbreCalibration(
                a=t.get("a", [1.0] * 12),
                b=t.get("b", [0.0] * 12),
            )
        
        # Loudness
        if "loudness" in data:
            l = data["loudness"]
            self.config.loudness = LoudnessCalibration(
                start_a=l.get("start", {}).get("a", 1.0),
                start_b=l.get("start", {}).get("b", 0.0),
                max_a=l.get("max", {}).get("a", 1.0),
                max_b=l.get("max", {}).get("b", 0.0),
            )
        
        # Confidence
        if "confidence" in data:
            c = data["confidence"]
            self.config.confidence = ConfidenceCalibration(
                source=c.get("source", [0.0, 1.0]),
                target=c.get("target", [0.0, 1.0]),
            )
        
        # Pitch
        if "pitch" in data:
            p = data["pitch"]
            self.config.pitch = PitchCalibration(
                power=p.get("power", 1.0),
                weights=p.get("weights", [1.0] * 12),
                normalize=p.get("normalize", "max"),
            )
        
        # Config section
        if "config" in data:
            cfg = data["config"]
            if "segmentation" in cfg:
                s = cfg["segmentation"]
                self.config.segmentation = SegmentationConfig(
                    min_segment_duration=s.get("min_segment_duration", 0.25),
                    novelty_smoothing=s.get("novelty_smoothing", 8),
                    peak_threshold=s.get("peak_threshold", 0.3),
                    peak_prominence=s.get("peak_prominence", 0.2),
                    max_segments_per_second=s.get("max_segments_per_second", 2.5),
                    beat_snap_tolerance=s.get("beat_snap_tolerance", 0.12),
                )
            if "features" in cfg:
                f = cfg["features"]
                self.config.features = FeatureConfig(
                    sample_rate=f.get("sample_rate", 44100),
                    frame_size=f.get("frame_size", 2048),
                    hop_size=f.get("hop_size", 512),
                )
            self.config.tatums_per_beat = cfg.get("tatums_per_beat", 2)
            self.config.time_signature = cfg.get("time_signature", 4)
    
    def calibrate_timbre(self, mfcc: np.ndarray) -> List[float]:
        """
        Apply affine transform to timbre vector.
        
        Args:
            mfcc: Raw MFCC coefficients (12 values)
            
        Returns:
            Calibrated timbre vector (12 values)
        """
        if len(mfcc) != 12:
            mfcc = np.pad(mfcc, (0, max(0, 12 - len(mfcc))))[:12]
        
        a = np.array(self.config.timbre.a[:12])
        b = np.array(self.config.timbre.b[:12])
        
        calibrated = mfcc * a + b
        return [round(float(v), 4) for v in calibrated]
    
    def calibrate_pitches(self, chroma: np.ndarray) -> List[float]:
        """
        Apply pitch calibration with weights, power transform, and normalization.
        
        Args:
            chroma: Raw chroma vector (12 values)
            
        Returns:
            Calibrated pitch vector (12 values, normalized to 0-1)
        """
        if len(chroma) != 12:
            chroma = np.pad(chroma, (0, max(0, 12 - len(chroma))))[:12]
        
        # Apply power transform
        chroma = np.power(np.clip(chroma, 0, None), self.config.pitch.power)
        
        # Apply weights
        weights = np.array(self.config.pitch.weights[:12])
        chroma = chroma * weights
        
        # Normalize
        if self.config.pitch.normalize == "l1":
            norm = np.sum(np.abs(chroma))
            if norm > 0:
                chroma = chroma / norm
        elif self.config.pitch.normalize == "l2":
            norm = np.linalg.norm(chroma)
            if norm > 0:
                chroma = chroma / norm
        else:  # "max" normalization
            max_val = np.max(chroma)
            if max_val > 0:
                chroma = chroma / max_val
        
        return [round(float(min(1.0, max(0.0, v))), 4) for v in chroma]
    
    def calibrate_loudness_start(self, db: float) -> float:
        """Apply loudness_start calibration."""
        return round(
            db * self.config.loudness.start_a + self.config.loudness.start_b,
            3
        )
    
    def calibrate_loudness_max(self, db: float) -> float:
        """Apply loudness_max calibration."""
        return round(
            db * self.config.loudness.max_a + self.config.loudness.max_b,
            3
        )
    
    def calibrate_confidence(self, conf: float) -> float:
        """
        Apply piecewise linear confidence remapping.
        
        Args:
            conf: Raw confidence value (0-1)
            
        Returns:
            Calibrated confidence value (0-1)
        """
        source = np.array(self.config.confidence.source)
        target = np.array(self.config.confidence.target)
        
        if len(source) < 2 or len(target) < 2:
            return conf
        
        calibrated = float(np.interp(conf, source, target))
        return round(min(1.0, max(0.0, calibrated)), 3)
    
    def get_segmentation_config(self) -> SegmentationConfig:
        """Get segmentation parameters."""
        return self.config.segmentation
    
    def get_feature_config(self) -> FeatureConfig:
        """Get feature extraction parameters."""
        return self.config.features
    
    def get_time_signature(self) -> int:
        """Get default time signature."""
        return self.config.time_signature
    
    def get_tatums_per_beat(self) -> int:
        """Get tatums per beat."""
        return self.config.tatums_per_beat
