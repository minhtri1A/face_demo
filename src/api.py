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



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc chỉ định ['http://localhost:3000']
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hàm khởi tạo ffmpeg để ghi HLS
def start_ffmpeg_hls_writer():
    os.makedirs("hls", exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", "640x480",  # chỉnh đúng với size frame gửi từ client
        "-r", "15",
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-f", "hls",
        "-hls_time", "2",
        "-hls_list_size", "5",
        "-hls_flags", "delete_segments",
        "./hls/playlist.m3u8"
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

# Khởi tạo ffmpeg từ đầu chương trình
ffmpeg_proc = start_ffmpeg_hls_writer()


# Mount /hls vao thu muc hls
app.mount("/hls", StaticFiles(directory="hls"), name="hls")

print('start api')

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
async def websocket_endpoint(websocket: WebSocket):
    print('start socket hls')
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            print('receive from client hls')

            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            print('check frame')
            if frame is None:
                continue

            # Resize cho đúng kích thước với ffmpeg
            frame = cv2.resize(frame, (640, 480))
            print('check resize')
            # Face detection
            faces = face_app.get(frame)
            for idx, face in enumerate(faces):
                print('face num ', idx)
                box = face.bbox.astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                for landmark in face.kps:
                    x, y = map(int, landmark)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                cv2.putText(frame, str(idx), (box[0], box[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 1)

            # Gửi về client ảnh base64 (nếu cần)
            # _, buffer = cv2.imencode('.jpg', frame)
            # encoded = base64.b64encode(buffer).decode('utf-8')
            # await websocket.send_text(encoded)
            # print('send to client')

            # Gửi frame vào ffmpeg để tạo stream HLS
            try:
                print('ffmpeg_proc to client')
                ffmpeg_proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("⚠️ ffmpeg stream closed unexpectedly.")
                break

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print("Error:", e)
    finally:
        try:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
        except:
            pass


# ngrok.set_auth_token("2whbuvHI5jH1j8avQ2PMHPwpdU3_3ofa364QXXiV4invKSoaq")
# public_url = ngrok.connect(8000)
# print("Public URL:", public_url)

# --- Chạy ứng dụng (khi chạy file trực tiếp) ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)  # Chạy với uvicorn






