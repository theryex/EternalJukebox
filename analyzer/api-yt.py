from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from grabber import JukeboxGrabber
from analyzer import FloppaAnalyzer
import uvicorn
import time
import os
import yt_dlp

app = FastAPI()
task_status = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

grabber = JukeboxGrabber()
analyzer = FloppaAnalyzer()

class AnalysisRequest(BaseModel):
    url: str
    override: str = None

def get_youtube_meta(url):
    """Helper to extract YouTube metadata without downloading the full file yet"""
    ydl_opts = {'quiet': True, 'no_warnings': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            'id': f"yt-{info['id']}",
            'name': info.get('title', 'Unknown YouTube Track'),
            'title': info.get('title', 'Unknown YouTube Track'),
            'artist': info.get('uploader', 'YouTube'),
            'duration': info.get('duration', 0) * 1000,
            'url': url,
            'source': 'youtube' # Flag to tell the processor how to handle it
        }

def run_full_process(track_id, meta):
    last_msg = ""

    def update_log(msg, prog=None):
        nonlocal last_msg
        if msg == last_msg: return
        current = task_status.get(track_id, {})
        new_prog = prog if prog is not None else current.get("progress", 0)
        task_status[track_id] = {"status": "processing", "progress": new_prog, "log": msg}
        last_msg = msg
        time.sleep(0.3)

    try:
        # 1. Grab Audio
        if meta.get('source') == 'youtube':
            update_log("Direct YouTube link detected. Downloading...", 10)
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': f'downloads/{track_id}.%(ext)s',
                'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([meta['url']])
            audio_file = f"downloads/{track_id}.wav"
            actual_title = meta['title']
        else:
            # Original Spotify workflow
            audio_file, actual_title = grabber.get_audio(meta, status_callback=lambda m: update_log(m, 10))
        
        # 2. Transition to CUDA
        update_log(f"Synced: {actual_title}. Engaging A5000 CUDA...", 40)

        # 3. Analyze with Verbose Callback
        analyzer.analyze(audio_file, track_id, meta, status_callback=lambda m, p: update_log(m, p))
        
        task_status[track_id] = {"status": "completed", "progress": 100, "log": "Success! Analysis deposited."}
    except Exception as e:
        task_status[track_id] = {"status": "error", "progress": 0, "log": f"Error: {str(e)}"}

@app.post("/analyze/")
async def start_analysis(req: AnalysisRequest, background_tasks: BackgroundTasks):
    # Determine if it's YouTube or Spotify
    is_youtube = "youtube.com" in req.url or "youtu.be" in req.url
    
    try:
        if is_youtube:
            meta = get_youtube_meta(req.url)
        else:
            meta = grabber.get_metadata(req.url)
            if not meta: raise Exception("Invalid Spotify URL")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Handle the manual override if provided
    if req.override:
        meta['name'] = req.override
        meta['title'] = req.override
        meta['artist'] = ""
    
    track_id = meta['id']
    task_status[track_id] = {"status": "queued", "progress": 0, "log": f"Waking up Floppa for {meta['title']}..."}
    background_tasks.add_task(run_full_process, track_id, meta)
    
    return {"task_id": track_id, "meta": meta}

@app.get("/analyze/status/{task_id}")
async def get_status(task_id: str):
    return task_status.get(task_id, {"status": "not_found", "progress": 0, "log": "Initializing..."})

if __name__ == "__main__":
    # Ensure downloads directory exists for YT audio
    if not os.path.exists("downloads"): os.makedirs("downloads")
    uvicorn.run(app, host="0.0.0.0", port=6874)
