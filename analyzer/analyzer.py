"""
Hybrid Audio Analyzer for EternalJukebox.

Produces high-quality audio analysis compatible with Spotify/EchoNest format using:
- Madmom neural network beat detection
- Essentia HPCP and MFCC extraction
- GPU acceleration where available
- Graceful CPU fallback

Output JSON matches the Spotify Audio Analysis API schema.
"""

import json
import os
import numpy as np
import ffmpeg
from typing import Optional, Callable, Dict, Any, List, Tuple
from dotenv import load_dotenv

from features_hybrid import HybridFeatureExtractor
from calibration import AnalysisCalibrator

load_dotenv()


class HybridAnalyzer:
    """
    Audio analyzer producing EchoNest-compatible analysis.
    
    Drop-in replacement for FloppaAnalyzer with improved:
    - Beat detection (Madmom RNN)
    - Timbre extraction (Essentia MFCC)
    - Pitch analysis (Essentia HPCP)
    - Structural sections (Laplacian eigenmaps)
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        calibration_path: Optional[str] = None,
        use_gpu: bool = True,
        prefer_essentia: bool = True,
        use_madmom_beats: bool = True,
    ):
        self.output_dir = output_dir or os.getenv("JUKEBOX_ANALYSIS_DIR", "./")
        
        # Calibration
        if calibration_path is None:
            calibration_path = os.path.join(
                os.path.dirname(__file__), "calibration.json"
            )
        self.calibrator = AnalysisCalibrator(calibration_path)
        
        # Feature extractor configuration
        self.use_gpu = use_gpu
        self.prefer_essentia = prefer_essentia
        self.use_madmom_beats = use_madmom_beats
        
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        print(f"--- Hybrid Analyzer initialized ---")
        print(f"--- Output: {self.output_dir} ---")
        print(f"--- GPU: {use_gpu}, Essentia: {prefer_essentia}, Madmom: {use_madmom_beats} ---")
    
    def load_audio_ffmpeg(
        self, file_path: str, target_sr: int = 22050
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Load audio file using FFmpeg."""
        try:
            out, _ = (
                ffmpeg
                .input(file_path)
                .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar=str(target_sr))
                .run(capture_stdout=True, capture_stderr=True)
            )
            return np.frombuffer(out, np.float32).copy(), target_sr
        except ffmpeg.Error as e:
            print(f"FFmpeg Error: {e.stderr.decode() if e.stderr else 'Unknown'}")
            return None, None
    
    def analyze(
        self,
        input_path: str,
        track_id: str,
        track_info: Optional[Dict[str, Any]] = None,
        status_callback: Optional[Callable[[str, int], None]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Analyze an audio file and produce EchoNest-compatible JSON.
        
        Args:
            input_path: Path to audio file
            track_id: Unique identifier for this track
            track_info: Optional dict with 'name', 'title', 'artist'
            status_callback: Optional progress callback(message, percent)
            
        Returns:
            Tuple of (analysis_dict, output_path)
        """
        track_info = track_info or {}
        
        def report(msg: str, prog: int) -> None:
            if status_callback:
                status_callback(msg, prog)
            print(f"[{prog}%] {msg}")
        
        # =====================================================================
        # 1. LOAD AUDIO
        # =====================================================================
        report("Loading audio with FFmpeg...", 10)
        y, sr = self.load_audio_ffmpeg(input_path)
        if y is None:
            return None, None
        
        duration = float(len(y) / sr)
        report(f"Audio loaded: {duration:.2f}s @ {sr}Hz", 20)
        
        # =====================================================================
        # 2. INITIALIZE HYBRID FEATURE EXTRACTOR
        # =====================================================================
        extractor = HybridFeatureExtractor(
            use_gpu=self.use_gpu,
            prefer_essentia=self.prefer_essentia,
            use_madmom_beats=self.use_madmom_beats,
            sample_rate=sr,
            hop_length=self.calibrator.get_feature_config().hop_size,
            frame_length=self.calibrator.get_feature_config().frame_size,
            n_mfcc=12,
            status_callback=report,
        )
        
        # =====================================================================
        # 3. BEAT DETECTION
        # =====================================================================
        tempo, beat_times, onset_env, downbeat_indices = extractor.compute_beats(y, sr)
        
        # Create beats array
        beats = self._create_beat_events(beat_times, onset_env, sr, duration)
        
        # Create tatums (subdivisions of beats)
        tatums = self._create_tatums(beat_times, duration)
        
        # Create bars (groups of beats based on downbeats)
        bars = self._create_bars(beat_times, downbeat_indices, duration)
        
        report(f"Rhythm: {len(beats)} beats, {len(bars)} bars, tempo={tempo:.1f} BPM", 55)
        
        # =====================================================================
        # 4. FEATURE EXTRACTION
        # =====================================================================
        mfcc = extractor.compute_mfcc(y, sr)
        chroma = extractor.compute_chroma(y, sr)
        rms_db = extractor.compute_rms_db(y, sr)
        
        report(f"Features extracted: MFCC {mfcc.shape}, Chroma {chroma.shape}", 70)
        
        # =====================================================================
        # 5. SEGMENT DETECTION
        # =====================================================================
        report("Computing segments...", 75)
        segments = self._compute_segments(
            y, sr, beat_times, mfcc, chroma, rms_db, duration
        )
        report(f"Segments: {len(segments)}", 80)
        
        # =====================================================================
        # 6. SECTION DETECTION
        # =====================================================================
        sections_raw = extractor.compute_sections(
            y, sr, beat_times, mfcc, chroma, duration
        )
        sections = self._enrich_sections(
            sections_raw, y, sr, onset_env, chroma, tempo
        )
        report(f"Sections: {len(sections)}", 90)
        
        # =====================================================================
        # 7. COMPUTE TRACK-LEVEL FEATURES
        # =====================================================================
        track_data = self._compute_track_features(
            y, sr, tempo, chroma, rms_db, duration, track_info
        )
        
        # =====================================================================
        # 8. ASSEMBLE FINAL JSON
        # =====================================================================
        report("Assembling final JSON...", 95)
        
        final_json = {
            "info": {
                "service": "SPOTIFY",
                "id": track_id,
                "name": track_info.get("name", "Unknown"),
                "title": track_info.get("title", track_info.get("name", "Unknown")),
                "artist": track_info.get("artist", "Unknown"),
                "duration": int(duration * 1000),  # milliseconds
            },
            "track": track_data,
            "bars": bars,
            "beats": beats,
            "tatums": tatums,
            "sections": sections,
            "segments": segments,
            "analysis": {
                "sections": sections,
                "bars": bars,
                "beats": beats,
                "segments": segments,
                "tatums": tatums,
            },
            "audio_summary": {
                "duration": round(duration, 4),
            },
            "meta": {
                "analyzer_version": "4.0.0-hybrid",
                "platform": "EternalJukebox",
                "detailed_status": "OK",
                "status_code": 0,
                "analysis_time": 0.0,
                "input_process": f"ffmpeg L+R {sr}->22050",
            },
        }
        
        # =====================================================================
        # 9. SAVE OUTPUT
        # =====================================================================
        dest_path = os.path.join(self.output_dir, f"{track_id}.json")
        report(f"Saving to: {dest_path}", 99)
        
        with open(dest_path, "w") as f:
            json.dump(final_json, f, default=lambda x: float(x))
        
        report("Analysis complete!", 100)
        return final_json, dest_path
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _create_beat_events(
        self,
        beat_times: np.ndarray,
        onset_env: np.ndarray,
        sr: int,
        duration: float,
    ) -> List[Dict[str, Any]]:
        """Create beat event list with confidence."""
        import librosa
        
        beats = []
        hop_length = self.calibrator.get_feature_config().hop_size
        
        for i, start in enumerate(beat_times):
            if i + 1 < len(beat_times):
                dur = float(beat_times[i + 1] - start)
            else:
                dur = min(60.0 / 120.0, duration - start)  # Default to 0.5s
            
            # Sample onset envelope for confidence
            frame = int(start * sr / hop_length)
            if frame < len(onset_env):
                conf = float(onset_env[frame] / (np.max(onset_env) + 1e-9))
            else:
                conf = 0.5
            
            conf = self.calibrator.calibrate_confidence(conf)
            
            beats.append({
                "start": round(float(start), 5),
                "duration": round(dur, 5),
                "confidence": round(conf, 3),
            })
        
        return beats
    
    def _create_tatums(
        self, beat_times: np.ndarray, duration: float
    ) -> List[Dict[str, Any]]:
        """Create tatum events (subdivisions of beats)."""
        tatums_per_beat = self.calibrator.get_tatums_per_beat()
        tatums = []
        
        for i, start in enumerate(beat_times):
            if i + 1 < len(beat_times):
                beat_dur = float(beat_times[i + 1] - start)
            else:
                beat_dur = min(0.5, duration - start)
            
            tatum_dur = beat_dur / tatums_per_beat
            
            for t in range(tatums_per_beat):
                tatum_start = start + t * tatum_dur
                if tatum_start >= duration:
                    break
                tatums.append({
                    "start": round(float(tatum_start), 5),
                    "duration": round(tatum_dur, 5),
                    "confidence": round(0.5 / (t + 1), 3),  # Decreasing confidence
                })
        
        return tatums
    
    def _create_bars(
        self,
        beat_times: np.ndarray,
        downbeat_indices: List[int],
        duration: float,
    ) -> List[Dict[str, Any]]:
        """Create bar events from downbeat indices."""
        bars = []
        time_sig = self.calibrator.get_time_signature()
        
        if not downbeat_indices:
            # Fallback: every 4 beats
            downbeat_indices = list(range(0, len(beat_times), time_sig))
        
        for i, db_idx in enumerate(downbeat_indices):
            if db_idx >= len(beat_times):
                continue
            
            start = float(beat_times[db_idx])
            
            if i + 1 < len(downbeat_indices) and downbeat_indices[i + 1] < len(beat_times):
                end = float(beat_times[downbeat_indices[i + 1]])
            else:
                end = duration
            
            bars.append({
                "start": round(start, 5),
                "duration": round(end - start, 5),
                "confidence": 0.8,
            })
        
        return bars
    
    def _compute_segments(
        self,
        y: np.ndarray,
        sr: int,
        beat_times: np.ndarray,
        mfcc: np.ndarray,
        chroma: np.ndarray,
        rms_db: np.ndarray,
        duration: float,
    ) -> List[Dict[str, Any]]:
        """Compute segment events with timbre and pitch data."""
        import librosa
        
        hop_length = self.calibrator.get_feature_config().hop_size
        seg_config = self.calibrator.get_segmentation_config()
        
        # Compute novelty curve
        mfcc_delta = np.mean(np.abs(librosa.feature.delta(mfcc)), axis=0)
        
        # Combine with onset envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        min_len = min(len(onset_env), len(mfcc_delta))
        
        combined_novelty = (
            onset_env[:min_len] / (np.max(onset_env) + 1e-9) +
            mfcc_delta[:min_len] / (np.max(mfcc_delta) + 1e-9)
        )
        
        # Smooth novelty
        if seg_config.novelty_smoothing > 1:
            kernel = np.ones(seg_config.novelty_smoothing) / seg_config.novelty_smoothing
            combined_novelty = np.convolve(combined_novelty, kernel, mode="same")
        
        # Detect segment boundaries
        peaks = librosa.util.peak_pick(
            combined_novelty,
            pre_max=int(seg_config.peak_threshold * 10),
            post_max=int(seg_config.peak_threshold * 10),
            pre_avg=int(seg_config.peak_threshold * 10),
            post_avg=int(seg_config.peak_threshold * 10),
            delta=seg_config.peak_prominence,
            wait=int(seg_config.min_segment_duration * sr / hop_length),
        )
        
        peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
        
        # Snap to beats if available
        if len(beat_times) > 0:
            peak_times = self._snap_to_nearest(peak_times, beat_times, seg_config.beat_snap_tolerance)
        
        # Create segment times
        seg_times = np.unique(np.concatenate([[0.0], peak_times, [duration]]))
        
        # Enforce minimum duration
        seg_times = self._enforce_min_duration(seg_times, seg_config.min_segment_duration)
        
        # Limit segment density
        max_segs = int(duration * seg_config.max_segments_per_second)
        if len(seg_times) - 1 > max_segs:
            min_dur = duration / max_segs
            seg_times = self._enforce_min_duration(seg_times, min_dur)
        
        # Build segments
        segments = []
        for i in range(len(seg_times) - 1):
            start = float(seg_times[i])
            end = float(seg_times[i + 1])
            if end <= start:
                continue
            
            # Frame indices
            start_frame = int(start * sr / hop_length)
            end_frame = int(end * sr / hop_length)
            start_frame = min(start_frame, mfcc.shape[1] - 1)
            end_frame = min(end_frame, mfcc.shape[1])
            
            # Extract segment features
            seg_mfcc = mfcc[:, start_frame:end_frame]
            seg_chroma = chroma[:, start_frame:end_frame]
            seg_rms = rms_db[start_frame:min(end_frame, len(rms_db))]
            
            # Average timbre
            if seg_mfcc.size > 0:
                timbre_raw = np.mean(seg_mfcc, axis=1)
            else:
                timbre_raw = np.zeros(12)
            timbre = self.calibrator.calibrate_timbre(timbre_raw)
            
            # Average pitch
            if seg_chroma.size > 0:
                pitch_raw = np.mean(seg_chroma, axis=1)
            else:
                pitch_raw = np.zeros(12)
            pitches = self.calibrator.calibrate_pitches(pitch_raw)
            
            # Loudness
            if seg_rms.size > 0:
                loudness_start = self.calibrator.calibrate_loudness_start(float(seg_rms[0]))
                loudness_max = self.calibrator.calibrate_loudness_max(float(np.max(seg_rms)))
                loudness_max_time = float(np.argmax(seg_rms)) * hop_length / sr
                loudness_end = float(seg_rms[-1])
            else:
                loudness_start = -60.0
                loudness_max = -60.0
                loudness_max_time = 0.0
                loudness_end = -60.0
            
            # Confidence from onset envelope
            if start_frame < len(onset_env):
                conf = float(onset_env[start_frame] / (np.max(onset_env) + 1e-9))
            else:
                conf = 0.5
            conf = self.calibrator.calibrate_confidence(conf)
            
            segments.append({
                "start": round(start, 5),
                "duration": round(end - start, 5),
                "confidence": round(conf, 3),
                "loudness_start": round(loudness_start, 3),
                "loudness_max": round(loudness_max, 3),
                "loudness_max_time": round(loudness_max_time, 5),
                "loudness_end": round(loudness_end, 3),
                "pitches": pitches,
                "timbre": timbre,
            })
        
        # Fix loudness_end chain
        for i in range(len(segments) - 1):
            segments[i]["loudness_end"] = segments[i + 1]["loudness_start"]
        
        return segments
    
    def _enrich_sections(
        self,
        sections_raw: List[Dict[str, Any]],
        y: np.ndarray,
        sr: int,
        onset_env: np.ndarray,
        chroma: np.ndarray,
        tempo: float,
    ) -> List[Dict[str, Any]]:
        """Add key, mode, tempo, loudness to sections."""
        import librosa
        
        hop_length = self.calibrator.get_feature_config().hop_size
        rms_db = 20.0 * np.log10(librosa.feature.rms(y=y)[0] + 1e-9)
        
        sections = []
        for sec in sections_raw:
            start = sec["start"]
            dur = sec["duration"]
            end = start + dur
            
            start_frame = int(start * sr / hop_length)
            end_frame = int(end * sr / hop_length)
            start_frame = min(start_frame, chroma.shape[1] - 1)
            end_frame = min(end_frame, chroma.shape[1])
            
            # Section chroma
            sec_chroma = chroma[:, start_frame:end_frame]
            if sec_chroma.size > 0:
                avg_chroma = np.mean(sec_chroma, axis=1)
            else:
                avg_chroma = np.zeros(12)
            
            # Key detection (simplified)
            key = int(np.argmax(avg_chroma))
            key_conf = float(avg_chroma[key] / (np.sum(avg_chroma) + 1e-9))
            
            # Mode detection (major/minor heuristic)
            major_weights = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
            minor_weights = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
            
            major_score = np.sum(np.roll(major_weights, key) * avg_chroma)
            minor_score = np.sum(np.roll(minor_weights, key) * avg_chroma)
            
            mode = 1 if major_score >= minor_score else 0
            mode_conf = abs(major_score - minor_score) / (major_score + minor_score + 1e-9)
            
            # Section tempo (use track tempo)
            sec_tempo = tempo
            tempo_conf = 0.7
            
            # Section loudness
            sec_rms = rms_db[start_frame:min(end_frame, len(rms_db))]
            loudness = float(np.mean(sec_rms)) if sec_rms.size > 0 else -60.0
            
            sections.append({
                "start": round(start, 5),
                "duration": round(dur, 5),
                "confidence": round(sec.get("confidence", 0.5), 3),
                "loudness": round(loudness, 3),
                "tempo": round(sec_tempo, 3),
                "tempo_confidence": round(tempo_conf, 3),
                "key": key,
                "key_confidence": round(key_conf, 3),
                "mode": mode,
                "mode_confidence": round(mode_conf, 3),
                "time_signature": self.calibrator.get_time_signature(),
                "time_signature_confidence": 0.8,
            })
        
        return sections
    
    def _compute_track_features(
        self,
        y: np.ndarray,
        sr: int,
        tempo: float,
        chroma: np.ndarray,
        rms_db: np.ndarray,
        duration: float,
        track_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute track-level summary features."""
        # Overall chroma
        avg_chroma = np.mean(chroma, axis=1) if chroma.size > 0 else np.zeros(12)
        
        # Key
        key = int(np.argmax(avg_chroma))
        key_conf = float(avg_chroma[key] / (np.sum(avg_chroma) + 1e-9))
        
        # Mode
        major_weights = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        minor_weights = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        major_score = np.sum(np.roll(major_weights, key) * avg_chroma)
        minor_score = np.sum(np.roll(minor_weights, key) * avg_chroma)
        mode = 1 if major_score >= minor_score else 0
        mode_conf = abs(major_score - minor_score) / (major_score + minor_score + 1e-9)
        
        # Loudness
        loudness = float(np.mean(rms_db)) if rms_db.size > 0 else -60.0
        
        return {
            "num_samples": len(y),
            "duration": round(duration, 5),
            "sample_md5": "",
            "offset_seconds": 0,
            "window_seconds": 0,
            "analysis_sample_rate": sr,
            "analysis_channels": 1,
            "end_of_fade_in": 0.0,
            "start_of_fade_out": round(duration * 0.95, 5),
            "loudness": round(loudness, 3),
            "tempo": round(tempo, 3),
            "tempo_confidence": 0.7,
            "time_signature": self.calibrator.get_time_signature(),
            "time_signature_confidence": 0.9,
            "key": key,
            "key_confidence": round(key_conf, 3),
            "mode": mode,
            "mode_confidence": round(mode_conf, 3),
            "codestring": "",
            "code_version": 3.15,
            "echoprintstring": "",
            "echoprint_version": 4.15,
            "synchstring": "",
            "synch_version": 1.0,
            "rhythmstring": "",
            "rhythm_version": 1.0,
        }
    
    def _snap_to_nearest(
        self, times: np.ndarray, targets: np.ndarray, tolerance: float
    ) -> np.ndarray:
        """Snap times to nearest target within tolerance."""
        snapped = []
        for t in times:
            diffs = np.abs(targets - t)
            min_idx = np.argmin(diffs)
            if diffs[min_idx] <= tolerance:
                snapped.append(targets[min_idx])
            else:
                snapped.append(t)
        return np.array(snapped)
    
    def _enforce_min_duration(
        self, times: np.ndarray, min_dur: float
    ) -> np.ndarray:
        """Remove times that create segments shorter than min_dur."""
        if len(times) < 2:
            return times
        
        result = [times[0]]
        for t in times[1:]:
            if t - result[-1] >= min_dur:
                result.append(t)
        
        # Ensure we include the end
        if result[-1] != times[-1]:
            if times[-1] - result[-1] >= min_dur:
                result.append(times[-1])
            else:
                result[-1] = times[-1]
        
        return np.array(result)


# Backward compatibility alias
FloppaAnalyzer = HybridAnalyzer


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyzer.py <audio_file> [track_id]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    track_id = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(input_file).rsplit(".", 1)[0]
    
    analyzer = HybridAnalyzer()
    result, output_path = analyzer.analyze(input_file, track_id)
    
    if result:
        print(f"\nAnalysis saved to: {output_path}")
        print(f"Beats: {len(result.get('beats', []))}")
        print(f"Bars: {len(result.get('bars', []))}")
        print(f"Sections: {len(result.get('sections', []))}")
        print(f"Segments: {len(result.get('segments', []))}")
    else:
        print("Analysis failed!")
        sys.exit(1)