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

print('*****start routes')

router = APIRouter()

# Instantiate the service
face_recognition_service = FaceRecognitionService()

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