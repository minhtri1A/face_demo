import os
import cv2
import base64
import asyncio
import uuid
import json
import time
import subprocess
import numpy as np

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from src.core.utils import start_ffmpeg_hls_writer, wait_for_hls_ready_socket
from src.core.config import HLS_DIR
from src.services.face_recognition import FaceRecognitionService


router = APIRouter()

# Instantiate the service
face_recognition_service = FaceRecognitionService()
face_app = face_recognition_service

@router.websocket("/ws/face")
async def websocket_endpoint(websocket: WebSocket):
    print('start socket')
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()  # Nhận bytes từ client
            print('receive from client')
            # Decode image từ bytes
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # Face detection 
            frame = await face_app.process_face_frame(frame)

            if frame is None:
                continue 

            # Encode lại frame thành JPEG để gửi về client
            _, buffer = cv2.imencode('.jpg', frame)
            encoded = base64.b64encode(buffer).decode('utf-8')

            await websocket.send_text(encoded)  # gửi lại base64 ảnh
            print('send to client')
    except WebSocketDisconnect:
        print("Client disconnected")

@router.websocket("/ws-hls/face")
async def websocket_hls_streaming(websocket: WebSocket):
    print('*****WebSocket /ws-hls/face connected.')
    await websocket.accept()

    stream_id = str(uuid.uuid4())
    ffmpeg_proc = None
    stream_path = os.path.join(HLS_DIR, stream_id) # stream_path
    countFrame = 0
  
    is_started = False
  
    try:
        ffmpeg_proc = start_ffmpeg_hls_writer(stream_id)

        print(f"*****HLS stream started for ID: {stream_id}")
        while True:
            data = await websocket.receive_bytes()

            # Check send stream_id to client
            if is_started == False:
                is_started = True
                response_object = {"HLS_STREAM_ID": stream_id}
                await websocket.send_text(json.dumps(response_object))

            #done
            if data == b"DONE":
                print('*****Done!!!')
                ffmpeg_proc.stdin.close()
                ffmpeg_proc.wait(timeout=5)
            
            # Get frame
            frame = await asyncio.to_thread(cv2.imdecode, np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue         

            # Face detection 
            frame = await face_app.process_face_frame(frame)

            if frame is None:
                continue 

            # Gửi frame vào ffmpeg để tạo stream HLS
            try:
                countFrame = countFrame +1
                print('*****Write from to hls', countFrame)
                ffmpeg_proc.stdin.write(frame.tobytes())
               
                # check file m3u8 created
                # if is_check_create_hls == False:
                #     is_check_create_hls = True
                #     playlist_file_path = os.path.join(stream_path, "playlist.m3u8")
                #     asyncio.create_task(wait_for_hls_ready_socket(playlist_file_path, stream_id, websocket))
            except BrokenPipeError:
                print(f"*****FFmpeg stream for ID {stream_id} closed unexpectedly.")
                break # Thoát vòng lặp nếu pipe bị hỏng

    except WebSocketDisconnect:
        print(f"*****Client disconnected from /ws-hls/face (ID: {stream_id})")
    except Exception as e:
        print(f"*****Error in /ws-hls/face (ID: {stream_id}): {e}")
    finally:
        print('*****finally')
        if ffmpeg_proc:
            try:
                ffmpeg_proc.stdin.close()
                ffmpeg_proc.wait(timeout=5) # Đợi FFmpeg kết thúc
                print(f"*****FFmpeg process for ID {stream_id} terminated.")
            except subprocess.TimeoutExpired:
                print(f"*****FFmpeg process for ID {stream_id} did not terminate, killing it.")
                ffmpeg_proc.kill()
            except Exception as e:
                print(f"*****Error while cleaning up FFmpeg process for ID {stream_id}: {e}")