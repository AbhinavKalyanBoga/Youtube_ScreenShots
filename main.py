import os
import streamlit as st
import yt_dlp
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import imagehash
import zipfile
from urllib.parse import urlparse, parse_qs

st.title("🎬 YouTube Playlist Frame Extractor & Deduplicator")

# --- Sidebar Input ---
st.sidebar.header("YouTube Playlist Settings")
url_input = st.sidebar.text_input("Paste YouTube Playlist or Video URL", "https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF")

# Determine type of URL
is_playlist = 'list=' in url_input and 'v=' not in url_input
is_video_with_playlist = 'list=' in url_input and 'v=' in url_input
is_single_video = 'list=' not in url_input and 'v=' in url_input

video_index = None
if is_playlist:
    video_index = st.sidebar.number_input("Video Number in Playlist (1-based)", min_value=1, step=1, value=1)

if "agreed" not in st.session_state:
    st.session_state.agreed = False
if "start_processing" not in st.session_state:
    st.session_state.start_processing = False

if not st.session_state.start_processing:
    if st.button("Run Extraction & Deduplication"):
        st.session_state.start_processing = True
        st.rerun()
    st.stop()

try:
    st.write("🔍 Resolving video URL...")

    parsed = urlparse(url_input)
    query = parse_qs(parsed.query)

    if is_video_with_playlist:
        video_id = query['v'][0]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        video_title = "selected_playlist_video"

    elif is_playlist:
        ydl_extract_opts = {
            'quiet': True,
            'extract_flat': True,
            'force_generic_extractor': False
        }
        with yt_dlp.YoutubeDL(ydl_extract_opts) as ydl:
            playlist_info = ydl.extract_info(url_input, download=False)
        if 'entries' not in playlist_info:
            raise ValueError("Could not extract entries from playlist.")
        if video_index is None or video_index > len(playlist_info['entries']):
            raise IndexError("Invalid video index for playlist.")
        entry = playlist_info['entries'][video_index - 1]
        video_url = f"https://www.youtube.com/watch?v={entry['id']}"
        video_title = entry['title']

    elif is_single_video:
        video_id = query['v'][0]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        video_title = "standalone_video"

    else:
        raise ValueError("Invalid YouTube URL. Please check your input.")

    # --- License & Permission Check ---
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)

    license_type = info_dict.get("license", "youtube")
    privacy_status = info_dict.get("privacy_status", "public")
    video_title = info_dict.get("title", video_title)

    st.write(f"🔍 Video: **{video_title}**")
    st.write(f"🔐 License: **{license_type}** | Privacy: **{privacy_status}**")

    if license_type != "creativeCommon" and not st.session_state.agreed:
        st.warning("⚠️ This video is not marked as Creative Commons. You must have permission to extract content.")
        if st.checkbox("✅ I confirm I have permission to process this video"):
            st.session_state.agreed = True
            st.rerun()
        st.stop()

    output_filename = "video.mp4"
    url_tracking_file = "last_url.txt"

    st.write(f"🎯 Video to download: {video_url}")

    should_download = True
    if os.path.exists(output_filename) and os.path.exists(url_tracking_file):
        with open(url_tracking_file, "r") as f:
            last_url = f.read().strip()
        if last_url == video_url:
            st.info("📂 Video already exists, skipping download")
            should_download = False
        else:
            os.remove(output_filename)
            st.info("♻️ URL changed, re-downloading video...")

    if should_download:
        st.write("⬇️ Downloading video...")
        ydl_download_opts = {
            'outtmpl': output_filename,
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'merge_output_format': 'mp4'
        }
        with yt_dlp.YoutubeDL(ydl_download_opts) as ydl:
            ydl.download([video_url])
        with open(url_tracking_file, "w") as f:
            f.write(video_url)
        st.success("✅ Video downloaded")

    # --- Frame Extraction ---
    output_dir = "scene_output"
    os.makedirs(output_dir, exist_ok=True)
    st.write("🎞️ Extracting frames using SSIM...")

    clip = VideoFileClip(output_filename)
    fps = clip.fps
    duration = clip.duration

    resize_shape = (320, 240)
    threshold = 0.75
    min_time_gap = 2.0
    last_saved_time = -min_time_gap
    scene_id = 0
    prev_gray = None
    sample_interval = 1.0

    st.write(f"🎥 Video length: {duration:.2f} seconds")
    progress_bar = st.progress(0)
    status_text = st.empty()

    timestamps = np.arange(0, duration, sample_interval)
    total_steps = len(timestamps)

    for i, t in enumerate(timestamps):
        frame_rgb = clip.get_frame(t)
        frame_resized = cv2.resize(frame_rgb, resize_shape)
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)

        if prev_gray is not None:
            score, _ = ssim(prev_gray, frame_gray, full=True)
            if score < threshold or (t - last_saved_time) >= min_time_gap:
                out_path = os.path.join(output_dir, f"scene_{scene_id:04d}.jpg")
                cv2.imwrite(out_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                scene_id += 1
                last_saved_time = t
        else:
            out_path = os.path.join(output_dir, f"scene_{scene_id:04d}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            scene_id += 1
            last_saved_time = t

        prev_gray = frame_gray
        progress_bar.progress((i + 1) / total_steps)
        status_text.text(f"Processing time {t:.1f}s / {duration:.1f}s")

    status_text.text("✅ Frame extraction completed.")
    st.success(f"✅ Extracted {scene_id} keyframes to '{output_dir}'")

    # --- Duplicate Removal ---
    st.write("🧹 Removing duplicate frames...")
    hashes = {}
    deleted = 0

    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        try:
            with Image.open(filepath) as img:
                hash_val = imagehash.phash(img)
                duplicate_found = False
                for existing_hash in hashes:
                    if abs(hash_val - existing_hash) <= 5:
                        os.remove(filepath)
                        deleted += 1
                        duplicate_found = True
                        break
                if not duplicate_found:
                    hashes[hash_val] = filename
        except Exception as e:
            st.warning(f"⚠️ Error reading {filename}: {e}")

    st.success(f"🗑️ Removed {deleted} duplicate frames.")

    # --- Zip and provide download ---
    zip_path = "deduplicated_frames.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=file)

    with open(zip_path, "rb") as f:
        st.download_button("📦 Download Deduplicated Frames (ZIP)", f, file_name="deduplicated_frames.zip")

    st.balloons()

except Exception as e:
    st.error(f"❌ Error: {e}")
