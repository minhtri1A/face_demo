import asyncio
import subprocess
import os
import json

from src.core.config import HLS_DIR

# Create ffmpeg to write hls
def start_ffmpeg_hls_writer(stream_id: str):
    stream_path = os.path.join(HLS_DIR, stream_id)
    os.makedirs(stream_path, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", "640x480",
        "-r", "15",
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-f", "hls",
        "-hls_time", "2",
        "-force_key_frames", "expr:gte(t,n_forced*2)",
        "-hls_list_size", "5",
        "-hls_flags", "delete_segments",
        os.path.join(stream_path, "playlist.m3u8")
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

#Check create m3u8
async def wait_for_hls_ready(playlist_path: str, stream_id: str, websocket):
    print(f"******Waiting create file m3u8")
    while not (os.path.exists(playlist_path)):
        await asyncio.sleep(0.5)  # check mỗi 500ms

    print(f"*****Create success file m3u8 - send steam_id to client")
  
    response_object = {"HLS_STREAM_ID": stream_id}
    await websocket.send_text(json.dumps(response_object))