import os
import cv2
import numpy as np
import base64
import asyncio
import uuid
import json
import time

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from src.core.utils import start_ffmpeg_hls_writer, wait_for_hls_ready
from src.core.config import HLS_DIR
from src.services.face_recognition import FaceRecognitionService, face_app
import subprocess

router = APIRouter()

# Instantiate the service
face_recognition_service = FaceRecognitionService(face_app)

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
            faces = face_app.get(frame)

            for idx, face in enumerate(faces):
                print('face ', idx)
                box = face.bbox.astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                for landmark in face.kps:
                    x, y = map(int, landmark)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
               
                cv2.putText(frame, str(idx), (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

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

    # Tạo một ID duy nhất cho stream HLS này
    is_check_create_hls = False
    stream_id = str(uuid.uuid4())
    ffmpeg_proc = None
    stream_path = os.path.join(HLS_DIR, stream_id) # Lấy đường dẫn stream_path ở đây
    countFrame = 0

    try:
        ffmpeg_proc = start_ffmpeg_hls_writer(stream_id)

        print(f"*****HLS stream started for ID: {stream_id}")
        while True:
            data = await websocket.receive_bytes()

            #done
            if data == b"DONE":
                print('*****Done!!!')
                ffmpeg_proc.stdin.close()
                ffmpeg_proc.wait(timeout=5)
            start1 = time.perf_counter()   
            
            # Get frame
            frame = await asyncio.to_thread(cv2.imdecode, np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            
            # Resize khung hình nếu cần (đảm bảo khớp với ffmpeg)
            if frame.shape[1] != 640 or frame.shape[0] != 480:
                frame = await asyncio.to_thread(cv2.resize, frame, (640, 480))
           

            # Face detection + embedding
            start2 = time.perf_counter()   
            faces = await asyncio.to_thread(face_app.get, frame)
            end2 = time.perf_counter()
            print("*****Thời gian chạy 2:", end2 - start2, "giây")

            # Face recognition
            if faces:
                embs = np.stack([face.embedding for face in faces], axis=0)
                results = face_recognition_service.find_best_match_batch(embs, threshold=0.4)
            else:
                results = []
            
            for idx, face in enumerate(faces):
                box = face.bbox.astype(int)
                face_name = results[idx]["name"]
                score = results[idx]["score"]
                # Draw rectangle
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                # Draw landmark
                for landmark in face.kps:
                    x, y = map(int, landmark)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                # Draw text
                cv2.putText(frame, f"{face_name} ({score:.2f})", (box[0], box[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
           
                            
            # Gửi frame vào ffmpeg để tạo stream HLS
            try:
                countFrame = countFrame +1
                print('*****Write from to hls', countFrame)
                ffmpeg_proc.stdin.write(frame.tobytes())

                # check file m3u8 created
                if is_check_create_hls == False:
                    is_check_create_hls = True
                    playlist_file_path = os.path.join(stream_path, "playlist.m3u8")
                    asyncio.create_task(wait_for_hls_ready(playlist_file_path, stream_id, websocket))
                end1 = time.perf_counter()
                print("*****Thời gian chạy 1:", end1 - start1, "giây")
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