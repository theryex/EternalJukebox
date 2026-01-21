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
        # Check if it's a YouTube URL
        youtube_patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in youtube_patterns:
            match = re.search(pattern, track_url)
            if match:
                video_id = match.group(1)
                # Fetch metadata from YouTube using yt-dlp
                try:
                    with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
                        info = ydl.extract_info(track_url, download=False)
                        return {
                            "id": f"yt-{video_id}",
                            "name": info.get('title', 'Unknown'),
                            "artist": info.get('uploader', 'Unknown'),
                            "album": "YouTube",
                            "title": info.get('title', 'Unknown'),
                            "isrc": None,
                            "duration_ms": int(info.get('duration', 0) * 1000),
                            "source": "youtube",
                            "youtube_url": track_url
                        }
                except Exception as e:
                    print(f"YouTube metadata extraction failed: {e}")
                    return None
        
        # Otherwise, treat as Spotify URL
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
                "duration_ms": t['duration_ms'],
                "source": "spotify"
            }
        except Exception as e:
            return None

    def get_audio(self, meta, status_callback=None):
        def report(msg):
            if status_callback: status_callback(msg)
            print(msg)

        track_id = meta['id']
        
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'outtmpl': os.path.join(self.audio_dir, f"{track_id}.%(ext)s"),
            'noplaylist': True, 'quiet': True, 'no_warnings': True,
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'm4a'}],
        }

        # If source is YouTube, download directly from the URL
        if meta.get('source') == 'youtube':
            report(f"Downloading from YouTube: {meta.get('title', 'Unknown')}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([meta['youtube_url']])
            return os.path.join(self.audio_dir, f"{track_id}.m4a"), meta.get('title', 'Unknown')
        
        # Otherwise, search for Spotify track on YouTube
        spotify_dur = meta['duration_ms'] / 1000.0
        
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