import librosa
import numpy as np
import json
import os
import ffmpeg
import torch
from nnAudio import features
from scipy.ndimage import median_filter
from dotenv import load_dotenv

load_dotenv()

class FloppaAnalyzer:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or os.getenv("JUKEBOX_ANALYSIS_DIR", "./")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_gpu = (self.device == "cuda")
        print(f"--- 2026 Hybrid Analyzer: {self.device.upper()} mode ---")
        print(f"--- Output directory: {self.output_dir} ---")
        
        if not os.path.exists(self.output_dir):
            print(f"--- Creating output directory: {self.output_dir} ---")
            os.makedirs(self.output_dir)

    def load_audio_ffmpeg(self, file_path):
        try:
            out, _ = (
                ffmpeg
                .input(file_path)
                .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar='22050')
                .run(capture_stdout=True, capture_stderr=True)
            )
            # .copy() ensures the array is writable for PyTorch
            return np.frombuffer(out, np.float32).copy(), 22050
        except ffmpeg.Error as e:
            print(f"FFmpeg Error: {e.stderr.decode() if e.stderr else 'Unknown'}")
            return None, None

    def analyze(self, input_path, track_id, track_info=None, status_callback=None):
        def report(msg, prog):
            if status_callback: status_callback(msg, prog)
            print(f"[{prog}%] {msg}")

        y, sr = self.load_audio_ffmpeg(input_path)
        if y is None: return None, None
        duration = float(len(y) / sr)
        
        report("Tracking Rhythmic Pulse and Snap Points...", 45)
        # ... (rest of beat logic)

        report(f"Engaging CUDA: Extracting Spectral Data (A5000)...", 60)
        # ... (rest of CUDA logic)

        report("Calculating Spectral Novelty & Boundary Points...", 75)
        # ... (rest of novelty logic)

        report("Processing pitches and timbre mapping...", 85)
        # ... (rest of segment logic)

        report("Identifying Structural Sections...", 95)
        # ... (rest of section logic)

        report("Finalizing JSON assembly...", 99)
        # ... (save logic)
        
        # --- 1. RHYTHM ANALYSIS ---
        report("Tracking Rhythmic Pulse and Snap Points...", 40)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
        tempo_raw, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, tightness=100)
        tempo = float(tempo_raw.item() if hasattr(tempo_raw, 'item') else tempo_raw)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        beats = []
        tatums = []
        for i in range(len(beat_times)):
            start = float(beat_times[i])
            dur = float(beat_times[i+1] - beat_times[i]) if i < len(beat_times)-1 else (60.0/tempo)
            conf = float(onset_env[min(beat_frames[i], len(onset_env)-1)] / (np.max(onset_env) + 1e-9))
            
            beats.append({"start": round(start, 5), "duration": round(dur, 5), "confidence": round(conf, 3)})
            tatums.append({"start": round(start, 5), "duration": round(dur/2, 5), "confidence": round(conf/2, 3)})
            tatums.append({"start": round(start + (dur/2), 5), "duration": round(dur/2, 5), "confidence": round(conf/4, 3)})

        # --- 2. FEATURE EXTRACTION ---
        cqt_np = None
        mfcc_np = None
        if self.use_gpu:
            try:
                report(f"Engaging CUDA: Extracting Spectral Data on {self.device.upper()}...", 55)
                y_torch = torch.from_numpy(y).to(self.device).float()
                cqt_layer = features.CQT2010v2(sr=sr, hop_length=512, n_bins=84).to(self.device)
                mfcc_layer = features.MFCC(sr=sr, hop_length=512, n_mfcc=12).to(self.device)
                with torch.no_grad():
                    audio_batch = y_torch.unsqueeze(0)
                    cqt_np = cqt_layer(audio_batch).squeeze(0).cpu().numpy()
                    mfcc_np = mfcc_layer(audio_batch).squeeze(0).cpu().numpy()
                del y_torch, cqt_layer, mfcc_layer
                torch.cuda.empty_cache()
            except Exception as e:
                report(f"CUDA Error: {e}. Falling back to CPU...", 55)
                self.use_gpu = False

        if cqt_np is None:
            cqt_np = np.abs(librosa.cqt(y=y, sr=sr, hop_length=512, n_bins=84))
            mfcc_np = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=12)

        # --- 3. NOVELTY-BASED SEGMENTATION ---
        report("Calculating Spectral Novelty & Boundary Points...", 75)
        mfcc_delta = np.mean(np.abs(librosa.feature.delta(mfcc_np)), axis=0)
        combined_novelty = (onset_env[:len(mfcc_delta)] / np.max(onset_env)) + (mfcc_delta / np.max(mfcc_delta))
        onsets = librosa.onset.onset_detect(onset_envelope=combined_novelty, sr=sr, units='frames', backtrack=True)
        
        bound_frames = np.unique(np.concatenate(([0], onsets, [mfcc_np.shape[-1] - 1])))
        rms = librosa.feature.rms(y=y)[0]

        report("Processing pitches and timbre mapping...", 85)
        segments_data = []
        for i in range(len(bound_frames)-1):
            s_idx, e_idx = int(bound_frames[i]), int(bound_frames[i+1])
            if s_idx >= e_idx: e_idx = s_idx + 1

            s_cqt = cqt_np[:, s_idx:e_idx]
            chroma = np.zeros(12)
            for b in range(s_cqt.shape[0]): chroma[b % 12] += np.mean(s_cqt[b, :])
            if np.max(chroma) > 0:
                chroma /= np.max(chroma)
                chroma[chroma < 0.4] = 0 
                chroma = np.power(chroma, 2.5) 

            s_mfcc = mfcc_np[:, s_idx:e_idx]
            mfcc_avg = np.mean(s_mfcc, axis=1)
            mfcc_avg[0] = (mfcc_avg[0] + 50) * 1.8 
            t_weights = np.array([1, 1.4, 1.2, 1.1, 1, 1, 1, 1, 1, 1, 1, 1])
            timbre = (mfcc_avg * t_weights).tolist()

            l_max = float(librosa.amplitude_to_db([np.max(rms[s_idx:e_idx]) if e_idx < len(rms) else 0])[0])

            segments_data.append({
                "start": round(float(librosa.frames_to_time(s_idx, sr=sr)), 5),
                "duration": round(float(librosa.frames_to_time(e_idx - s_idx, sr=sr)), 5),
                "confidence": 0.5, "loudness_start": -60.0, "loudness_max": round(l_max, 3),
                "loudness_max_time": 0.02, "loudness_end": 0.0,
                "pitches": [round(float(x), 4) for x in chroma], "timbre": [round(float(x), 4) for x in timbre]
            })

        # --- 4. STRUCTURAL SECTIONS ---
        report("Identifying Structural Sections...", 95)
        boundary_frames = librosa.segment.agglomerative(librosa.feature.stack_memory(mfcc_np, n_steps=10), 5)
        boundary_times = librosa.frames_to_time(boundary_frames, sr=sr)
        sections = [{"start": round(float(boundary_times[i]), 5), "duration": round(float(boundary_times[i+1]-boundary_times[i]), 5), "confidence": 1.0, "loudness": -7.0, "tempo": round(tempo, 3), "key": 0, "mode": 1, "time_signature": 4} for i in range(len(boundary_times)-1)]

        report("Finalizing JSON assembly...", 99)
        final_json = {
            "info": {"service": "SPOTIFY", "id": track_id, "name": track_info.get('name', 'Unknown'), "title": track_info.get('title', 'Unknown'), "artist": track_info.get('artist', 'Unknown'), "duration": int(duration * 1000)},
            "analysis": {"sections": sections, "bars": [beats[i] for i in range(0, len(beats), 4)], "beats": beats, "segments": segments_data, "tatums": tatums},
            "audio_summary": {"duration": round(duration, 4)}
        }

        dest_path = os.path.join(self.output_dir, f"{track_id}.json")
        print(f"--- Saving analysis to: {dest_path} ---")
        with open(dest_path, "w") as f:
            # Force NumPy types to float during dump
            json.dump(final_json, f, default=lambda x: float(x))
        print(f"--- Analysis saved successfully! ---")
        
        return final_json, dest_path