import os
os.environ.pop("MPLBACKEND", None)
import matplotlib
matplotlib.use("Agg")


import cv2
import numpy as np
import base64
import asyncio
from fastapi import FastAPI, WebSocket
from insightface.app import FaceAnalysis
from starlette.websockets import WebSocketDisconnect
from pyngrok import ngrok
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import subprocess
import uuid
import json



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc chỉ định ['http://localhost:3000']
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thư mục chứa các stream HLS riêng biệt
HLS_DIR = "hls_streams"
os.makedirs(HLS_DIR, exist_ok=True)

# Mount thư mục /hls_streams
app.mount(f"/{HLS_DIR}", StaticFiles(directory=HLS_DIR), name="hls_streams")


# Hàm khởi tạo ffmpeg để ghi HLS cho mỗi client
def start_ffmpeg_hls_writer(stream_id: str):
    stream_path = os.path.join(HLS_DIR, stream_id)
    os.makedirs(stream_path, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", "640x480",  # Đảm bảo khớp với kích thước khung hình từ client
        "-r", "15",
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-f", "hls",
        "-hls_time", "2",
        "-hls_list_size", "5",
        "-hls_flags", "delete_segments",
        os.path.join(stream_path, "playlist.m3u8")
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

# Khởi tạo ffmpeg từ đầu chương trình
# ffmpeg_proc = start_ffmpeg_hls_writer()

print('*****Start api')

# --- Khởi tạo insightface ---
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

@app.websocket("/ws/face")
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
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 1)

            # Encode lại frame thành JPEG để gửi về client
            _, buffer = cv2.imencode('.jpg', frame)
            encoded = base64.b64encode(buffer).decode('utf-8')

            await websocket.send_text(encoded)  # gửi lại base64 ảnh
            print('send to client')
    except WebSocketDisconnect:
        print("Client disconnected")


@app.websocket("/ws-hls/face")
async def websocket_hls_streaming(websocket: WebSocket):
    print('*****WebSocket /ws-hls/face connected.')
    await websocket.accept()

    # Tạo một ID duy nhất cho stream HLS này
    stream_id = str(uuid.uuid4())
    ffmpeg_proc = None
    try:
        ffmpeg_proc = start_ffmpeg_hls_writer(stream_id)
        # Gửi lại ID stream HLS cho client để họ có thể truy cập playlist.m3u8
        response_object = {"HLS_STREAM_ID": stream_id}
        await websocket.send_text(json.dumps(response_object))
        print(f"*****HLS stream started for ID: {stream_id}")

        while True:
            data = await websocket.receive_bytes()
            frame = await asyncio.to_thread(cv2.imdecode, np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

            if frame is None:
                continue
            print('*****Receive fram from client')

            # Resize khung hình nếu cần (đảm bảo khớp với ffmpeg)
            if frame.shape[1] != 640 or frame.shape[0] != 480:
                frame = await asyncio.to_thread(cv2.resize, frame, (640, 480))

            # Xử lý khuôn mặt
            faces = await asyncio.to_thread(face_app.get, frame)
            for idx, face in enumerate(faces):
                print('*****face detection num ', idx)
                box = face.bbox.astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                for landmark in face.kps:
                    x, y = map(int, landmark)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                cv2.putText(frame, str(idx), (box[0], box[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 1)

            # Gửi frame vào ffmpeg để tạo stream HLS
            try:
                print('write from to hls')
                ffmpeg_proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print(f"*****FFmpeg stream for ID {stream_id} closed unexpectedly.")
                break # Thoát vòng lặp nếu pipe bị hỏng

    except WebSocketDisconnect:
        print(f"*****Client disconnected from /ws-hls/face (ID: {stream_id})")
    except Exception as e:
        print(f"*****Error in /ws-hls/face (ID: {stream_id}): {e}")
    finally:
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
        # Bạn có thể thêm logic để xóa thư mục stream HLS của client đó nếu muốn
        # import shutil
        # if os.path.exists(os.path.join(HLS_DIR, stream_id)):
        #     shutil.rmtree(os.path.join(HLS_DIR, stream_id))


# ngrok.set_auth_token("2whbuvHI5jH1j8avQ2PMHPwpdU3_3ofa364QXXiV4invKSoaq")
# public_url = ngrok.connect(8000)
# print("Public URL:", public_url)

# --- Chạy ứng dụng (khi chạy file trực tiếp) ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)  # Chạy với uvicorn






