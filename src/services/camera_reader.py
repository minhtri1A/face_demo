import cv2
import asyncio
import time

import numpy as np
from src.core.utils import start_ffmpeg_hls_writer

class RTSPCameraReader:
    def __init__(self, rtsp_url: str, stream_id: str, face_recognition_service, ):
        self.rtsp_url = rtsp_url
        self.stream_id = stream_id
        self.face_recognition_service = face_recognition_service
        self.video_capture = None
        self.running = False
        self.ffmpeg_process = None

    async def start_reading(self):
        print(f"*****Starting RTSP stream capture for {self.rtsp_url} (ID: {self.stream_id})")
        self.running = True
        self.video_capture = cv2.VideoCapture(self.rtsp_url)

        if not self.video_capture.isOpened():
            print(f"*****Error: Could not open RTSP stream {self.rtsp_url}")
            self.running = False
            return

        # Khởi tạo tiến trình FFmpeg để ghi HLS
        self.ffmpeg_process = start_ffmpeg_hls_writer(self.stream_id)
        print(f"*****HLS stream writer started for ID: {self.stream_id}")

        frame_count = 0
        while self.running:

            ret, frame = self.video_capture.read()

            if not ret:
                print(f"*****Error: Could not read frame from {self.rtsp_url}. Reconnecting...")
                self.video_capture.release()
                await asyncio.sleep(2) # Đợi 2 giây trước khi thử kết nối lại
                self.video_capture = cv2.VideoCapture(self.rtsp_url)
                if not self.video_capture.isOpened():
                    print(f"*****Failed to reconnect to {self.rtsp_url}. Stopping.")
                    self.running = False
                continue

            # Face detection 
            frame = await self.face_recognition_service.process_face_frame(frame)
            
            if frame is None:
                continue 

            #Wirte frame to hls
            try:
                self.ffmpeg_process.stdin.write(frame.tobytes())
                frame_count += 1
                print('*****Write from to hls', frame_count)

            except BrokenPipeError:
                print(f"*****FFmpeg process for ID {self.stream_id} pipe broken. Restarting FFmpeg.")
                self.stop_reading() 
                await asyncio.sleep(1)
                await self.start_reading() 
                break 
        
        print(f"*****Stopping RTSP stream capture for {self.rtsp_url} (ID: {self.stream_id})")
        self.stop_reading()

    def stop_reading(self):
        self.running = False
        if self.video_capture:
            self.video_capture.release()
            print(f"*****RTSP VideoCapture released for {self.rtsp_url}")
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.wait(timeout=5)
                print(f"*****FFmpeg process for ID {self.stream_id} terminated.")
            except Exception as e:
                print(f"*****Error stopping FFmpeg process for ID {self.stream_id}: {e}")