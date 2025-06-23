import os
os.environ.pop("MPLBACKEND", None)
import matplotlib
matplotlib.use("Agg")


import cv2
import numpy as np
import base64
import asyncio
from fastapi import FastAPI, WebSocket, HTTPException
from pydantic import BaseModel
from insightface.app import FaceAnalysis
from starlette.websockets import WebSocketDisconnect
from pyngrok import ngrok
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import subprocess
import uuid
import json
import asyncio
import time


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

#-----init

face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FACEBANK_DIR = os.path.join(SCRIPT_DIR, "facebank") 

class User(BaseModel):
    id: str
    name: str
    email: str = None

    @classmethod
    def as_form(
        cls,
        id: str = Form(...),
        name: str = Form(...),
        email: str = Form(None),
    ):
        return cls(id=id, name=name, email=email) 


# ---- healper
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
        "-force_key_frames", "expr:gte(t,n_forced*2)",
        "-hls_list_size", "5",
        "-hls_flags", "delete_segments",
        os.path.join(stream_path, "playlist.m3u8")
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

# Save face

# Save facebank
def save_facebank_append(new_embeddings: np.ndarray, new_names: list):
    # Nếu file đã tồn tại → load
    if os.path.exists("facebank_embeddings.npy") and os.path.exists("facebank_names.npy"):
        old_embeddings = np.load("facebank_embeddings.npy")
        old_names = np.load("facebank_names.npy")

        all_embeddings = np.concatenate([old_embeddings, new_embeddings], axis=0)
        all_names = np.concatenate([old_names, new_names], axis=0)
    else:
        all_embeddings = new_embeddings
        all_names = np.array(new_names)

    # Ghi lại file đã nối
    np.save("facebank_embeddings.npy", all_embeddings)
    np.save("facebank_names.npy", all_names)

print('*****Start api')

# --- API
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

# Save user API
@app.post("/save-face-user")
async def save_user_info(
    user: User = Depends(User.as_form),
    files: List[UploadFile] = File(...)
):

    print('****user data ', user_json)
    print('****file data ', user)

    return

    names = []
    embeddings = []

    # create folder by user.id
    user_folder = os.path.join(f'{UPLOAD_DIR}/images', user.id)
    os.makedirs(user_folder, exist_ok=True)

    # save image & embedding
    embs = []

    for file in files:
        iamgename = f"{len(os.listdir(user_folder)) + 1}.jpg"
        iamge_path = os.path.join(user_folder, iamgename)

        # Save image
        with open(iamge_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

            img = cv2.imread(file.file)
            if img is None:
                continue

        # Read image cv2
        img = cv2.imread(image_path)
        if img is None:
            print(f"[WARN] Không đọc được ảnh: {image_path}")
            continue

        # Get face embedding
        faces = app.get(img)
        if not faces:
            print(f"[WARN] Không tìm thấy khuôn mặt: {image_path}")
            continue
        emb = faces[0].embedding
        embs.append(emb)

    # Check embedding
    if embs:
        mean_emb = np.mean(embs, axis=0)
        embeddings.append(mean_emb)
        names.append(user_id)
        print(f"[OK] Thêm vào facebank: {user_id} ({len(embs)} ảnh)")

    #save info
    info_file = os.path.join(user_folder, "info.txt")
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(user.dict(), f, ensure_ascii=False, indent=4)

    #save image embedding to facebank
    save_facebank_append(np.array(embeddings), names)

    return f"Lưu thông tin user {user.name} thành công!!!"
        


# hls


async def wait_for_hls_ready(playlist_path: str, stream_id: str, websocket):
    print(f"******Waiting create file m3u8")
    while not (os.path.exists(playlist_path)):
        await asyncio.sleep(0.5)  # check mỗi 500ms

    print(f"*****Create success file m3u8 - send steam_id to client")
  
    response_object = {"HLS_STREAM_ID": stream_id}
    await websocket.send_text(json.dumps(response_object))


@app.websocket("/ws-hls/face")
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

            # Get frame
            frame = await asyncio.to_thread(cv2.imdecode, np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            
            # Resize khung hình nếu cần (đảm bảo khớp với ffmpeg)
            if frame.shape[1] != 640 or frame.shape[0] != 480:
                frame = await asyncio.to_thread(cv2.resize, frame, (640, 480))
               
            # Xử lý khuôn mặt
            faces = await asyncio.to_thread(face_app.get, frame)
            
            for idx, face in enumerate(faces):
                # print('*****face detection num ', idx)
                box = face.bbox.astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                for landmark in face.kps:
                    x, y = map(int, landmark)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                cv2.putText(frame, str(idx), (box[0], box[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 1)
           
                            
            # Gửi frame vào ffmpeg để tạo stream HLS
            try:
              # write frame to hls
                countFrame = countFrame +1
                print('*****Write from to hls', countFrame)
                ffmpeg_proc.stdin.write(frame.tobytes())

                # check file m3u8 created
                if is_check_create_hls == False:
                    is_check_create_hls = True
                    playlist_file_path = os.path.join(stream_path, "playlist.m3u8")
                    asyncio.create_task(wait_for_hls_ready(playlist_file_path, stream_id, websocket))
               
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
        # Bạn có thể thêm logic để xóa thư mục stream HLS của client đó nếu muốn
        import shutil
        if os.path.exists(stream_path):
            shutil.rmtree(stream_path)


# ngrok.set_auth_token("2whbuvHI5jH1j8avQ2PMHPwpdU3_3ofa364QXXiV4invKSoaq")
# public_url = ngrok.connect(8000)
# print("Public URL:", public_url)

# --- Chạy ứng dụng (khi chạy file trực tiếp) ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)  # Chạy với uvicorn






