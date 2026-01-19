import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import yt_dlp
import hashlib
import re
import time
from dotenv import load_dotenv

load_dotenv()

class JukeboxGrabber:
    def __init__(self):
        auth_manager = SpotifyClientCredentials()
        self.sp = spotipy.Spotify(auth_manager=auth_manager)
        self.audio_dir = os.getenv("JUKEBOX_AUDIO_DIR", "/mnt/ai_drive/EternalJukebox/data/audio/")

    def get_metadata(self, track_url):
        try:
            track_id = track_url.split("track/")[1].split("?")[0]
            t = self.sp.track(track_id)
            return {
                "id": track_id,
                "name": t['name'],
                "artist": t['artists'][0]['name'],
                "album": t['album']['name'],
                "title": f"{t['artists'][0]['name']} - {t['name']}",
                "isrc": t.get('external_ids', {}).get('isrc'),
                "duration_ms": t['duration_ms']
            }
        except Exception as e:
            return None

    def get_audio(self, meta, status_callback=None):
        def report(msg):
            if status_callback: status_callback(msg)
            print(msg)

        track_id = meta['id']
        spotify_dur = meta['duration_ms'] / 1000.0
        
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'outtmpl': os.path.join(self.audio_dir, f"{track_id}.%(ext)s"),
            'noplaylist': True, 'quiet': True, 'no_warnings': True,
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'm4a'}],
        }

        clean_name = re.sub(r'[?\"\'\(\)]', '', meta['name'])
        clean_artist = re.sub(r'[?\"\'\(\)]', '', meta['artist'])
        
        queries = []
        if meta.get('isrc'): queries.append(f"isrc:{meta['isrc']}")
        queries.append(f'"{clean_artist}" "{clean_name}" audio')
        queries.append(f"{clean_artist} {clean_name} official audio")

        best_q = None
        video_title = "Unknown Source"

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for q in queries:
                report(f"Searching: {q}")
                try:
                    res = ydl.extract_info(f"ytsearch1:{q}", download=False)
                    if 'entries' in res and len(res['entries']) > 0:
                        match = res['entries'][0]
                        yt_dur = match.get('duration', 0)
                        video_title = match.get('title', 'Unknown')
                        
                        if abs(yt_dur - spotify_dur) < 15: # 15s leeway
                            report(f"Match Confirmed: {video_title} ({yt_dur}s)")
                            best_q = q
                            break
                        else:
                            report(f"Skipping '{video_title}': Duration mismatch ({yt_dur}s)")
                except Exception as e:
                    continue

            if not best_q:
                report("Strict matching failed. Downloading best available result...")
                best_q = queries[-1]

            ydl.download([f"ytsearch1:{best_q}"])
            return os.path.join(self.audio_dir, f"{track_id}.m4a"), video_title