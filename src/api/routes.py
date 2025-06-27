import os
import json
import shutil
import numpy as np
import cv2
from typing import List

from fastapi import APIRouter, HTTPException, Form, Depends, UploadFile, File

from src.models.user import User
from src.services.face_recognition import FaceRecognitionService
from src.core.config import FACEBANK_DIR
from src.core.models.rtsp import RTSPStreamRequest

print('*****start routes')

router = APIRouter()

# Instantiate the service
face_recognition_service = FaceRecognitionService()
face_app = face_recognition_service.face_analysis_app


#-----RTSP camera API

# List active camera
active_rtsp_streams = {}

@router.post('/start-camera-rtsp')
async def start_camera_rtsp(request: RTSPStreamRequest):
  rtsp_url = request.rtsp_url
  stream_id = request.stream_id if request.stream_id else str(uuid.uuid4())

  if stream_id in active_rtsp_streams:
    print(f'***** Stream with ID {stream_id} is already running!!')
    return {"message": f"Stream with ID {stream_id} is already running."}

  is_check_create_hls = False
  ffmpeg_proc = None
  stream_path = os.path.join(HLS_DIR, stream_id) # stream_path
  countFrame = 0

  # read rtsp camera
  video_capture = cv2.VideoCapture(rtsp_url)
  if not video_capture.isOpened():
      print(f"*****Error: Could not open RTSP stream {rtsp_url}")
      return

  active_rtsp_streams[stream_id] = video_capture

  try:
      ffmpeg_proc = start_ffmpeg_hls_writer(stream_id)
      
      while True:
          ret, frame = video_capture.read()
          if not ret:
            print(f"*****Error: Could not read frame from {rtsp_url}. Reconnecting...")
            video_capture.release()
            asyncio.sleep(2)
            video_capture = cv2.VideoCapture(rtsp_url)
            if not video_capture.isOpened():
              print(f"*****Failed to reconnect to {rtsp_url}. Stopping.")
              # running = False
            continue

          # Resize khung hình (đảm bảo khớp với ffmpeg)
          if frame.shape[1] != 640 or frame.shape[0] != 480:
              frame = await asyncio.to_thread(cv2.resize, frame, (640, 480))

          # Face detection + embedding
          faces = await asyncio.to_thread(face_app.get, frame)

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
              # if is_check_create_hls == False:
              #     is_check_create_hls = True
              #     playlist_file_path = os.path.join(stream_path, "playlist.m3u8")
              #     asyncio.create_task(wait_for_hls_ready(playlist_file_path, stream_id, websocket))
          except BrokenPipeError:
                print(f"*****FFmpeg stream for ID {stream_id} closed unexpectedly.")
                break # Thoát vòng lặp nếu pipe bị hỏng

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
#Face bank API
@router.post("/save-user-to-facebank")
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
    face_recognition_service.save_facebank_append(np.array(embeddings), names)

    return f"*****Lưu thông tin user {user.name} thành công!!!"

@router.get("/load-user-from-facebank")
async def load_user_from_facebank():
  names, _ = face_recognition_service.get_facebank() # Ensure you're getting both names and embeddings or just names if that's all you need
  print('names facebak ', names)
  return {'names_in_facebank': [name.dict() for name in names]} # Assuming 'name' in facebank is a User object