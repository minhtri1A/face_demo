import os
os.environ.pop("MPLBACKEND", None)
import matplotlib
matplotlib.use("Agg")


import cv2
import numpy as np
import base64
import asyncio
from fastapi import FastAPI, WebSocket, HTTPException, Form, Depends, UploadFile, File
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
from typing import List
import shutil
from numpy.linalg import norm


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
FACEBANK_EMBEDDINGS_DIR = f"{FACEBANK_DIR}/facebank_embeddings.npy"
FACEBANK_NAMES_DIR = f"{FACEBANK_DIR}/facebank_names.npy"
FACEBANK_CACHE = {}


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

# facebank
def get_facebank():
    global FACEBANK_CACHE

    if not FACEBANK_CACHE:
        FACEBANK_CACHE["embeddings"] = np.load(FACEBANK_EMBEDDINGS_DIR)
        FACEBANK_CACHE["names"] = np.load(FACEBANK_NAMES_DIR, allow_pickle=True)

    return FACEBANK_CACHE["embeddings"], FACEBANK_CACHE["names"]

def reset_facebank_cache():
    global FACEBANK_CACHE
    FACEBANK_CACHE.clear()

# Get user info best match from facebank_embeddings
def find_best_match(embedding):
    facebank_embeddings, names = get_facebank()
    similarities = np.dot(facebank_embeddings, embedding) / (
        norm(facebank_embeddings, axis=1) * norm(embedding) + 1e-6
    )
    idx = np.argmax(similarities)
    return idx, similarities[idx], names[idx]

# Get user info best match multiple embebding from facebank_embeddings
def find_best_match_batch(embeddings: np.ndarray, threshold: float = 0.4):
    """
    So sánh nhiều embedding với facebank để tìm người giống nhất.

    Args:
        embeddings (np.ndarray): Mảng shape (N, 512) chứa embedding của nhiều khuôn mặt.
        threshold (float): Ngưỡng similarity để chấp nhận nhận diện.

    Returns:
        List[dict]: Danh sách kết quả. Mỗi kết quả là dict:
            {
                "index": index trong facebank,
                "score": cosine similarity,
                "name": tên người hoặc "unknown"
            }
    """
    facebank_embeddings, facebank_names = get_facebank()

    # Tính cosine similarity (N, M): N là số khuôn mặt, M là số người trong facebank
    sim_matrix = np.dot(embeddings, facebank_embeddings.T) / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) * 
        np.linalg.norm(facebank_embeddings, axis=1) + 1e-6
    )

    best_idxs = np.argmax(sim_matrix, axis=1)
    best_scores = np.max(sim_matrix, axis=1)

    results = []
    for i in range(len(embeddings)):
        idx = best_idxs[i]
        score = best_scores[i]
        name = facebank_names[idx]["name"] if score >= threshold else "unknown"
        results.append({
            "index": idx,
            "score": float(score),
            "name": name
        })
    
    return results

# Save facebank
def save_facebank_append(new_embeddings: np.ndarray, new_names: list):
    # check exists file
    if os.path.exists(FACEBANK_EMBEDDINGS_DIR) and os.path.exists(FACEBANK_NAMES_DIR):
        old_embeddings, old_names = get_facebank()
        all_embeddings = np.concatenate([old_embeddings, new_embeddings], axis=0)
        all_names = np.concatenate([old_names, new_names], axis=0).astype(object)
    else:
        all_embeddings = new_embeddings
        all_names = np.array(new_names)

    # save file
    np.save(FACEBANK_EMBEDDINGS_DIR, all_embeddings)
    np.save(FACEBANK_NAMES_DIR, all_names)

    # reset facebank
    reset_facebank_cache()

#Check create m3u8
async def wait_for_hls_ready(playlist_path: str, stream_id: str, websocket):
    print(f"******Waiting create file m3u8")
    while not (os.path.exists(playlist_path)):
        await asyncio.sleep(0.5)  # check mỗi 500ms

    print(f"*****Create success file m3u8 - send steam_id to client")
  
    response_object = {"HLS_STREAM_ID": stream_id}
    await websocket.send_text(json.dumps(response_object))

print('*****Start api')

# ------------- API
#Face demo API
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
@app.post("/save-user-to-facebank")
async def save_user_info(
    user: User = Depends(User.as_form),
    files: List[UploadFile] = File(...)
):

    print('****user data ', user)
    print('****file data ', files)

    names = []
    embeddings = []

    # create folder by user.id
    user_folder = os.path.join(f'{FACEBANK_DIR}/images', user.id)
    os.makedirs(user_folder, exist_ok=True)

    # save image & embedding
    embs = []

    for file in files:
        iamgename = f"{len(os.listdir(user_folder)) + 1}.jpg"
        image_path = os.path.join(user_folder, iamgename)

        # Save image
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read image cv2
        img = cv2.imread(image_path)
        if img is None:
            print(f"[WARN] Không đọc được ảnh: {image_path}")
            continue

        # Get face embedding
        faces = face_app.get(img)
        if not faces:
            print(f"[WARN] Không tìm thấy khuôn mặt: {image_path}")
            continue
        emb = faces[0].embedding
        embs.append(emb)

    # Check embedding
    if embs:
        mean_emb = np.mean(embs, axis=0)
        embeddings.append(mean_emb)
        names.append(user)
        print(f"[OK] Thêm vào facebank: {names} ({len(embs)} ảnh)")

    #save info
    info_file = os.path.join(user_folder, "info.txt")
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(user.dict(), f, ensure_ascii=False, indent=4)

    #save image embedding to facebank
    save_facebank_append(np.array(embeddings), names)

    return f"*****Lưu thông tin user {user.name} thành công!!!"

@app.get("/load-user-from-facebank")
async def load_user_from_facebank():
  names = np.load(FACEBANK_NAMES_DIR, allow_pickle=True)
  print('names facebak ', names)
  return 'hihi'      


# Face with HLS API
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
            start1 = time.perf_counter()   

            # Face detection
            faces = await asyncio.to_thread(face_app.get, frame)

            # Face recognition
            #--Gộp tất cả embedding của các face
            embs = np.stack([face.embedding for face in faces], axis=0)
            #--Gọi hàm batch
            results = find_best_match_batch(embs, threshold=0.4)


            for idx, face in enumerate(faces):
                # print('*****face detection num ', idx)
                box = face.bbox.astype(int)
                embs = face.embedding
                face_name = results[idx]["name"]
                score = results[idx]["score"]
                print(f'score:{score} - index: {idx} - face_name_recognition: {face_name}')
                # Draw rectangle
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                # Draw landmark
                for landmark in face.kps:
                    x, y = map(int, landmark)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                # Draw text
                cv2.putText(frame, f"{face_name} ({score:.2f})", (box[0], box[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 1)
            end1 = time.perf_counter()
            print("*****Thời gian chạy 1:", end1 - start1, "giây")
                            
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
        # clear hls stream
        # import shutil
        # if os.path.exists(stream_path):
        #     shutil.rmtree(stream_path)


# ngrok.set_auth_token("2whbuvHI5jH1j8avQ2PMHPwpdU3_3ofa364QXXiV4invKSoaq")
# public_url = ngrok.connect(8000)
# print("Public URL:", public_url)

# --- Chạy ứng dụng (khi chạy file trực tiếp) ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)  # Chạy với uvicorn






