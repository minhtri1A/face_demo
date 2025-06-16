import os
# Xử lý lỗi matplotlib backend trong môi trường headless
os.environ.pop("MPLBACKEND", None)
import matplotlib
matplotlib.use("Agg")

import cv2
import numpy as np
from insightface.app import FaceAnalysis



# Khởi tạo FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Hoặc ['CUDAExecutionProvider'] nếu có GPU
app.prepare(ctx_id=0)

# Mở video
current_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(current_dir, "video_face.mp4")
cap = cv2.VideoCapture(video_path)
print('cap.isOpened() ', cap.isOpened())

# Tạo thư mục lưu frame đã xử lý
os.makedirs("output_video", exist_ok=True)

# --- Kiểm tra video ---
if not cap.isOpened():
    print("Không thể mở video")
    exit()

# --- Lấy thông tin video ---
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path =f"output_video/output_video.mp4"

# --- Khởi tạo VideoWriter ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Face detection
    faces = app.get(frame)

    # draw
    for face in faces:
        box = face.bbox.astype(int)
        # ve khung
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # ve cac diem tren mat
        for landmark in face.kps:
            x, y = map(int, landmark)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        # them text cho khung
        if hasattr(face, 'embedding'):
            cv2.putText(frame, "Vua Dung Cu", (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 1)

    # Lưu frame đã xử lý mỗi 10 frame
    # print('fram_idx ', frame_idx)
    # if frame_idx % 5 == 0:
    #     output_path = f"output_frames/frame_{frame_idx:04d}.jpg"
    #     # cv2.imwrite(output_path, frame)
    #     print(f"Saved: {output_path}")

    # save video
    print("write frame ", frame_idx)
    out.write(frame)
    frame_idx += 1

cap.release()
print("✅ Hoàn tất xử lý video.")