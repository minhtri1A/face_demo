import os
import onnxruntime as ort

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

HLS_DIR = os.path.join(SCRIPT_DIR, "hls_streams")  
FACEBANK_DIR = os.path.join(SCRIPT_DIR, "facebank") 
FACEBANK_EMBEDDINGS_DIR = f"{FACEBANK_DIR}/facebank_embeddings.npy"
FACEBANK_NAMES_DIR = f"{FACEBANK_DIR}/facebank_names.npy"

# For insightface, you might consider moving the init here or into a service
# This is a good place to put global configurations like available providers
# print('*****check ort',ort.get_available_providers())