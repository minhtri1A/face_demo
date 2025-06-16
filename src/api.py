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



app = FastAPI()

# --- Khởi tạo insightface ---
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

@app.websocket("/ws/face")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()  # Nhận bytes từ client

            # Decode image từ bytes
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # Face detection
            faces = face_app.get(frame)
            for face in faces:
                box = face.bbox.astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                for landmark in face.kps:
                    x, y = map(int, landmark)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            # Encode lại frame thành JPEG để gửi về client
            _, buffer = cv2.imencode('.jpg', frame)
            encoded = base64.b64encode(buffer).decode('utf-8')

            await websocket.send_text(encoded)  # gửi lại base64 ảnh
    except WebSocketDisconnect:
        print("Client disconnected")


ngrok.set_auth_token("2whbuvHI5jH1j8avQ2PMHPwpdU3_3ofa364QXXiV4invKSoaq")
public_url = ngrok.connect(8000)
print("Public URL:", public_url)

# --- Chạy ứng dụng (khi chạy file trực tiếp) ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)  # Chạy với uvicorn






