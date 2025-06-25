import os
os.environ.pop("MPLBACKEND", None)
import matplotlib
matplotlib.use("Agg")

import sys
sys.path.append('/content/face_demo')
print("🛣 sys.path:", sys.path)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.api.routes import router as http_router
from src.api.websockets import router as websocket_router
from src.core.config import HLS_DIR
print('*****start api')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ['http://localhost:3000']
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Thư mục chứa các stream HLS riêng biệt
os.makedirs(HLS_DIR, exist_ok=True)

# Mount thư mục /hls_streams
app.mount(f"/{HLS_DIR}", StaticFiles(directory=HLS_DIR), name="hls_streams")

# Include routers
app.include_router(http_router)
app.include_router(websocket_router)

# --- Chạy ứng dụng (khi chạy file trực tiếp) ---
if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)