import streamlit as st
import os
import time
from grabber import JukeboxGrabber
from analyzer import FloppaAnalyzer
from dotenv import load_dotenv

load_dotenv()
grabber = JukeboxGrabber()
analyzer = FloppaAnalyzer()

st.set_page_config(page_title="Floppa Jukebox", page_icon="🎷")
st.sidebar.title("Server Monitor")
st.sidebar.success(f"GPU Engine: {analyzer.device.upper()} (A5000)")

st.title("🎷 Floppa Jukebox Analyzer")

url = st.text_input("Spotify Track URL")
override = st.text_input("Manual Search Override (Optional)", placeholder="e.g. Artist Name - Track Title")

if st.button("Process & Analyze"):
    if url:
        with st.status("Crunching Data...", expanded=True) as status:
            st.write("Fetching Spotify Metadata...")
            meta = grabber.get_metadata(url)
            
            if meta:
                # If the user provided an override, use that for the audio hunt
                if override:
                    meta['name'] = override
                    meta['artist'] = ""
                    meta['isrc'] = None # Kill ISRC if we are overriding
                
                st.write(f"Hunting for: **{meta['title']}**")
                audio_file = grabber.get_audio(meta)
                
                st.write("GPU Analysis in progress...")
                start_time = time.time()
                json_data, saved_path = analyzer.analyze(audio_file, meta['id'], meta)
                
                if json_data:
                    status.update(label=f"Done in {round(time.time()-start_time, 2)}s!", state="complete")
                    st.success(f"Track ID `{meta['id']}` ready for Jukebox.")
                    st.balloons()
