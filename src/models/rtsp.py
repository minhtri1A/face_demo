from pydantic import BaseModel
from typing import Optional

class RTSPStreamRequest(BaseModel):
    rtsp_url: str
    stream_id: Optional[str] = None
