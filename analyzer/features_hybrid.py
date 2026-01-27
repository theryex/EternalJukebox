"""
Hybrid feature extraction combining GPU acceleration with Essentia/Madmom quality.

This module provides a unified interface for audio feature extraction with:
- Madmom for neural network-based beat/downbeat detection (best quality)
- Essentia for HPCP, MFCC, and other audio features (EchoNest-compatible)
- GPU acceleration via torchaudio/nnAudio where available
- Graceful CPU fallback when GPU or specific libraries are unavailable

Author: Ryex (via AI assistant)
License: MIT
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Callable

# Library availability flags
HAS_MADMOM = False
HAS_ESSENTIA = False
HAS_TORCH = False
HAS_NNAUDIO = False

try:
    import madmom
    HAS_MADMOM = True
except ImportError:
    pass

try:
    import essentia.standard as es
    HAS_ESSENTIA = True
except ImportError:
    pass

try:
    import torch
    import torchaudio
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    pass

try:
    from nnAudio import features as nn_features
    HAS_NNAUDIO = True
except ImportError:
    pass

# Always available
import librosa
from scipy.ndimage import median_filter


class HybridFeatureExtractor:
    """
    Multi-library feature extractor with GPU acceleration and CPU fallback.
    
    Priority order for each feature:
    - Beats: Madmom RNN > librosa beat_track
    - MFCC: Essentia > GPU torchaudio > librosa
    - Chroma: Essentia HPCP > GPU CQT > librosa chroma_cqt
    - Sections: Laplacian eigenmaps on self-similarity matrix
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        prefer_essentia: bool = True,
        use_madmom_beats: bool = True,
        sample_rate: int = 22050,
        hop_length: int = 512,
        frame_length: int = 2048,
        n_mfcc: int = 12,
        status_callback: Optional[Callable[[str, int], None]] = None,
    ):
        self.use_gpu = use_gpu and HAS_TORCH
        self.prefer_essentia = prefer_essentia and HAS_ESSENTIA
        self.use_madmom_beats = use_madmom_beats and HAS_MADMOM
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.n_mfcc = n_mfcc
        self.status_callback = status_callback
        
        # GPU device
        self.device = "cuda" if self.use_gpu else "cpu"
        
        # Report available backends
        backends = []
        if self.use_madmom_beats:
            backends.append("Madmom")
        if self.prefer_essentia:
            backends.append("Essentia")
        if self.use_gpu:
            backends.append(f"GPU({self.device})")
        backends.append("librosa")
        
        self._report(f"Hybrid analyzer initialized: {', '.join(backends)}", 5)
    
    def _report(self, msg: str, progress: int) -> None:
        """Report progress to callback if available."""
        if self.status_callback:
            self.status_callback(msg, progress)
        print(f"[{progress}%] {msg}")
    
    # =========================================================================
    # BEAT DETECTION
    # =========================================================================
    
    def compute_beats(
        self, y: np.ndarray, sr: int
    ) -> Tuple[float, np.ndarray, np.ndarray, List[int]]:
        """
        Compute beat times and tempo.
        
        Returns:
            Tuple of (tempo, beat_times, onset_envelope, downbeat_indices)
        """
        if self.use_madmom_beats:
            return self._compute_beats_madmom(y, sr)
        return self._compute_beats_librosa(y, sr)
    
    def _compute_beats_madmom(
        self, y: np.ndarray, sr: int
    ) -> Tuple[float, np.ndarray, np.ndarray, List[int]]:
        """Neural network-based beat detection using Madmom."""
        self._report("Computing beats with Madmom RNN...", 40)
        
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # Madmom expects 44.1kHz
        madmom_sr = 44100
        if sr != madmom_sr:
            y_madmom = librosa.resample(y, orig_sr=sr, target_sr=madmom_sr)
        else:
            y_madmom = y
        
        fps = 100
        try:
            proc = madmom.features.DBNDownBeatTrackingProcessor(
                beats_per_bar=[3, 4], fps=fps
            )
            act = madmom.features.RNNDownBeatProcessor(fps=fps)(y_madmom)
            downbeats = proc(act)
            
            if downbeats.size == 0:
                raise RuntimeError("Madmom returned no beats")
            
            beat_times = downbeats[:, 0].astype(float)
            beat_positions = downbeats[:, 1].astype(int)  # 1 = downbeat, 2/3/4 = other beats
            
            # Find downbeat indices (where position == 1)
            downbeat_indices = np.where(beat_positions == 1)[0].tolist()
            
            # Compute onset envelope for confidence scoring
            onset_env = librosa.onset.onset_strength(
                y=y, sr=sr, hop_length=self.hop_length
            )
            
            # Estimate tempo from beat times
            if len(beat_times) >= 2:
                beat_diffs = np.diff(beat_times)
                median_ibi = np.median(beat_diffs)
                tempo = 60.0 / median_ibi if median_ibi > 0 else 120.0
            else:
                tempo = 120.0
            
            self._report(f"Madmom: {len(beat_times)} beats, tempo={tempo:.1f} BPM", 50)
            return tempo, beat_times, onset_env, downbeat_indices
            
        except Exception as e:
            self._report(f"Madmom failed: {e}, falling back to librosa", 45)
            return self._compute_beats_librosa(y, sr)
    
    def _compute_beats_librosa(
        self, y: np.ndarray, sr: int
    ) -> Tuple[float, np.ndarray, np.ndarray, List[int]]:
        """Standard librosa beat detection."""
        self._report("Computing beats with librosa...", 40)
        
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=self.hop_length
        )
        tempo_raw, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=self.hop_length
        )
        tempo = float(tempo_raw.item() if hasattr(tempo_raw, 'item') else tempo_raw)
        beat_times = librosa.frames_to_time(
            beat_frames, sr=sr, hop_length=self.hop_length
        )
        
        # Estimate downbeats every 4 beats (assuming 4/4 time)
        downbeat_indices = list(range(0, len(beat_times), 4))
        
        self._report(f"librosa: {len(beat_times)} beats, tempo={tempo:.1f} BPM", 50)
        return tempo, beat_times, onset_env, downbeat_indices
    
    # =========================================================================
    # MFCC / TIMBRE
    # =========================================================================
    
    def compute_mfcc(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute MFCC coefficients.
        
        Returns:
            MFCC matrix of shape (n_mfcc, n_frames)
        """
        if self.prefer_essentia:
            return self._compute_mfcc_essentia(y, sr)
        if self.use_gpu:
            return self._compute_mfcc_gpu(y, sr)
        return self._compute_mfcc_librosa(y, sr)
    
    def _compute_mfcc_essentia(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Essentia-based MFCC extraction."""
        self._report("Computing MFCC with Essentia...", 55)
        
        window = es.Windowing(type="hann")
        spectrum = es.Spectrum(size=self.frame_length)
        mfcc = es.MFCC(
            highFrequencyBound=min(11025, sr // 2),
            numberCoefficients=self.n_mfcc,
            inputSize=self.frame_length // 2 + 1
        )
        
        mfccs = []
        for start in range(0, max(len(y) - self.frame_length, 0) + 1, self.hop_length):
            frame = y[start:start + self.frame_length]
            if len(frame) < self.frame_length:
                frame = np.pad(frame, (0, self.frame_length - len(frame)), mode="constant")
            windowed = window(frame.astype(np.float32))
            spec = spectrum(windowed)
            _, mfcc_coeffs = mfcc(spec)
            mfccs.append(mfcc_coeffs)
        
        return np.array(mfccs).T  # (n_mfcc, n_frames)
    
    def _compute_mfcc_gpu(self, y: np.ndarray, sr: int) -> np.ndarray:
        """GPU-accelerated MFCC using torchaudio."""
        self._report(f"Computing MFCC with GPU ({self.device})...", 55)
        
        y_tensor = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).to(self.device)
        
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.frame_length,
                'hop_length': self.hop_length,
                'n_mels': 40,
            }
        ).to(self.device)
        
        with torch.no_grad():
            mfcc_tensor = mfcc_transform(y_tensor)
        
        return mfcc_tensor.squeeze(0).cpu().numpy()  # (n_mfcc, n_frames)
    
    def _compute_mfcc_librosa(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Fallback librosa MFCC."""
        self._report("Computing MFCC with librosa...", 55)
        return librosa.feature.mfcc(
            y=y, sr=sr,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.frame_length
        )
    
    # =========================================================================
    # CHROMA / PITCHES
    # =========================================================================
    
    def compute_chroma(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute chroma/pitch features.
        
        Returns:
            Chroma matrix of shape (12, n_frames)
        """
        if self.prefer_essentia:
            return self._compute_chroma_essentia(y, sr)
        if self.use_gpu and HAS_NNAUDIO:
            return self._compute_chroma_gpu(y, sr)
        return self._compute_chroma_librosa(y, sr)
    
    def _compute_chroma_essentia(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Essentia HPCP (Harmonic Pitch Class Profile) - most EchoNest-compatible."""
        self._report("Computing chroma with Essentia HPCP...", 60)
        
        window = es.Windowing(type="hann")
        spectrum = es.Spectrum(size=self.frame_length)
        spectral_peaks = es.SpectralPeaks(
            orderBy="magnitude",
            magnitudeThreshold=1e-6,
            sampleRate=float(sr)
        )
        hpcp = es.HPCP(size=12, sampleRate=float(sr))
        
        hpcps = []
        for start in range(0, max(len(y) - self.frame_length, 0) + 1, self.hop_length):
            frame = y[start:start + self.frame_length]
            if len(frame) < self.frame_length:
                frame = np.pad(frame, (0, self.frame_length - len(frame)), mode="constant")
            windowed = window(frame.astype(np.float32))
            spec = spectrum(windowed)
            freqs, mags = spectral_peaks(spec)
            hpcp_vec = hpcp(freqs, mags)
            hpcps.append(hpcp_vec)
        
        return np.array(hpcps).T  # (12, n_frames)
    
    def _compute_chroma_gpu(self, y: np.ndarray, sr: int) -> np.ndarray:
        """GPU-accelerated CQT-based chroma using nnAudio."""
        self._report(f"Computing chroma with GPU CQT ({self.device})...", 60)
        
        y_tensor = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).to(self.device)
        
        cqt_layer = nn_features.CQT2010v2(
            sr=sr, hop_length=self.hop_length, n_bins=84
        ).to(self.device)
        
        with torch.no_grad():
            cqt = cqt_layer(y_tensor).squeeze(0).cpu().numpy()
        
        # Fold CQT into chroma (84 bins -> 12 pitch classes)
        chroma = np.zeros((12, cqt.shape[1]))
        for b in range(cqt.shape[0]):
            chroma[b % 12] += cqt[b]
        
        # Normalize
        max_val = np.max(chroma, axis=0, keepdims=True)
        chroma = chroma / (max_val + 1e-9)
        
        return chroma
    
    def _compute_chroma_librosa(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Fallback librosa chroma_cqt."""
        self._report("Computing chroma with librosa...", 60)
        return librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=self.hop_length
        )
    
    # =========================================================================
    # LOUDNESS / RMS
    # =========================================================================
    
    def compute_rms_db(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Compute RMS energy in dB per frame."""
        rms = librosa.feature.rms(
            y=y, frame_length=self.frame_length, hop_length=self.hop_length
        )[0]
        return 20.0 * np.log10(rms + 1e-9)
    
    # =========================================================================
    # SECTIONS (Laplacian Eigenmaps)
    # =========================================================================
    
    def compute_sections(
        self,
        y: np.ndarray,
        sr: int,
        beat_times: np.ndarray,
        mfcc: np.ndarray,
        chroma: np.ndarray,
        duration: float,
        target_section_count: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compute structural sections using Laplacian eigenmaps.
        
        This approach uses self-similarity matrices from chroma and MFCC
        features to identify major structural boundaries.
        """
        self._report("Computing sections with Laplacian eigenmaps...", 85)
        
        if len(beat_times) < 4:
            # Fallback: divide into 4 sections
            section_times = np.linspace(0, duration, 5)
            return self._sections_from_times(section_times, duration)
        
        try:
            import scipy.sparse.csgraph
            import scipy.linalg
            from scipy.cluster.vq import kmeans2
        except ImportError:
            self._report("scipy not available for Laplacian, using simple sections", 85)
            return self._sections_fallback(y, sr, beat_times, duration)
        
        # Beat-synchronize features
        beat_frames = librosa.time_to_frames(
            beat_times, sr=sr, hop_length=self.hop_length
        )
        beat_frames = np.clip(beat_frames, 0, chroma.shape[1] - 1)
        beat_frames = np.unique(beat_frames)
        
        if len(beat_frames) < 4:
            return self._sections_fallback(y, sr, beat_times, duration)
        
        # Beat-sync chroma
        chroma_sync = self._beat_sync_mean(chroma, beat_frames)
        
        # Self-similarity matrix
        R = self._cosine_similarity_matrix(chroma_sync)
        Rf = median_filter(R, size=(1, 7))
        
        # Path similarity from MFCC
        mfcc_sync = self._beat_sync_mean(mfcc, beat_frames)
        path_distance = np.sum(np.diff(mfcc_sync, axis=1) ** 2, axis=0)
        sigma = max(float(np.median(path_distance)), 1e-9)
        path_sim = np.exp(-path_distance / sigma)
        R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)
        
        # Combine recurrence and path similarity
        deg_path = np.sum(R_path, axis=1)
        deg_rec = np.sum(Rf, axis=1)
        denom = np.sum((deg_path + deg_rec) ** 2)
        mu = float(deg_path.dot(deg_path + deg_rec) / denom) if denom > 0 else 0.5
        A = mu * Rf + (1.0 - mu) * R_path
        
        # Laplacian eigenvectors
        L = scipy.sparse.csgraph.laplacian(A, normed=True)
        _, evecs = scipy.linalg.eigh(L)
        evecs = median_filter(evecs, size=(9, 1))
        Cnorm = np.cumsum(evecs ** 2, axis=1) ** 0.5
        
        # Number of sections
        if target_section_count:
            k = target_section_count
        else:
            k = max(2, int(round(len(beat_times) / 32)))  # ~1 section per 32 beats
        k = min(k, 10, evecs.shape[1], len(beat_times) - 1)
        k = max(k, 2)
        
        # K-means clustering
        X = evecs[:, :k] / (Cnorm[:, k - 1:k] + 1e-9)
        np.random.seed(0)
        _, seg_ids = kmeans2(X, k, minit="points", iter=20)
        
        # Find section boundaries
        changes = np.flatnonzero(np.diff(seg_ids)) + 1
        section_times = np.concatenate([[0.0], beat_times[changes], [duration]])
        section_times = np.unique(section_times)
        
        return self._sections_from_times(section_times, duration)
    
    def _sections_fallback(
        self, y: np.ndarray, sr: int, beat_times: np.ndarray, duration: float
    ) -> List[Dict[str, Any]]:
        """Simple novelty-based section detection fallback."""
        self._report("Using novelty-based section detection fallback...", 85)
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=self.hop_length, n_mfcc=12)
        
        # MFCC delta novelty
        mfcc_delta = np.mean(np.abs(librosa.feature.delta(mfcc)), axis=0)
        combined = onset_env[:len(mfcc_delta)] / (np.max(onset_env) + 1e-9)
        combined += mfcc_delta / (np.max(mfcc_delta) + 1e-9)
        
        # Find peaks
        peaks = librosa.util.peak_pick(
            combined, pre_max=10, post_max=10,
            pre_avg=10, post_avg=10, delta=0.3, wait=50
        )
        
        peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=self.hop_length)
        section_times = np.concatenate([[0.0], peak_times, [duration]])
        section_times = np.unique(section_times)
        
        return self._sections_from_times(section_times, duration)
    
    def _sections_from_times(
        self, section_times: np.ndarray, duration: float
    ) -> List[Dict[str, Any]]:
        """Convert section boundary times to section dicts."""
        sections = []
        for i in range(len(section_times) - 1):
            start = float(section_times[i])
            end = float(section_times[i + 1])
            if end <= start:
                continue
            sections.append({
                "start": start,
                "duration": end - start,
                "confidence": 0.5,
            })
        return sections
    
    def _beat_sync_mean(self, features: np.ndarray, beat_frames: np.ndarray) -> np.ndarray:
        """Compute beat-synchronized mean of features."""
        n_beats = len(beat_frames)
        if n_beats == 0:
            return features
        
        synced = np.zeros((features.shape[0], n_beats))
        for i in range(n_beats):
            start = int(beat_frames[i])
            end = int(beat_frames[i + 1]) if i + 1 < n_beats else features.shape[1]
            if end > start:
                synced[:, i] = np.mean(features[:, start:end], axis=1)
        
        return synced
    
    def _cosine_similarity_matrix(self, features: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix."""
        norms = np.linalg.norm(features, axis=0, keepdims=True)
        normalized = features / (norms + 1e-9)
        return normalized.T @ normalized
