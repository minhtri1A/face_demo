import os
import numpy as np
from numpy.linalg import norm
import onnxruntime as ort

from insightface.app import FaceAnalysis
from src.core.config import FACEBANK_EMBEDDINGS_DIR, FACEBANK_NAMES_DIR

# insightFace: CUDAExecutionProvider CPUExecutionProvider
# This should be initialized once at startup
face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'], allowed_modules=["detection", "recognition"])
face_app.prepare(ctx_id=0)

FACEBANK_CACHE = {}

class FaceRecognitionService:
    def __init__(self, face_analysis_app: FaceAnalysis):
        self.face_analysis_app = face_analysis_app
        self._load_facebank_cache()

    def _load_facebank_cache(self):
        global FACEBANK_CACHE
        if os.path.exists(FACEBANK_EMBEDDINGS_DIR) and os.path.exists(FACEBANK_NAMES_DIR):
            FACEBANK_CACHE["embeddings"] = np.load(FACEBANK_EMBEDDINGS_DIR)
            FACEBANK_CACHE["names"] = np.load(FACEBANK_NAMES_DIR, allow_pickle=True)
        else:
            FACEBANK_CACHE["embeddings"] = np.array([])
            FACEBANK_CACHE["names"] = np.array([])
            print("[INFO] Facebank files not found. Initializing empty facebank.")

    def get_facebank(self):
        # Refresh cache if necessary or just return current
        if not FACEBANK_CACHE.get("embeddings") or not FACEBANK_CACHE.get("names"):
            self._load_facebank_cache()
        return FACEBANK_CACHE["embeddings"], FACEBANK_CACHE["names"]

    def reset_facebank_cache(self):
        global FACEBANK_CACHE
        FACEBANK_CACHE.clear()
        self._load_facebank_cache() # Reload after clearing

    def find_best_match(self, embedding):
        facebank_embeddings, names = self.get_facebank()
        if facebank_embeddings.size == 0:
            return None, 0.0, "unknown"

        similarities = np.dot(facebank_embeddings, embedding) / (
            norm(facebank_embeddings, axis=1) * norm(embedding) + 1e-6
        )
        idx = np.argmax(similarities)
        return idx, similarities[idx], names[idx]

    def find_best_match_batch(self, embeddings: np.ndarray, threshold: float = 0.4):
        facebank_embeddings, facebank_names = self.get_facebank()
        
        if facebank_embeddings.size == 0 or embeddings.size == 0:
            return [{"index": -1, "score": 0.0, "name": "unknown"}] * len(embeddings)


        sim_matrix = np.dot(embeddings, facebank_embeddings.T) / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) * np.linalg.norm(facebank_embeddings, axis=1) + 1e-6
        )

        best_idxs = np.argmax(sim_matrix, axis=1)
        best_scores = np.max(sim_matrix, axis=1)

        results = []
        for i in range(len(embeddings)):
            idx = best_idxs[i]
            score = best_scores[i]
            # Ensure names are handled correctly if they are Pydantic models
            name_obj = facebank_names[idx]
            name = name_obj.name if score >= threshold and hasattr(name_obj, 'name') else "unknown"
            results.append({
                "index": int(idx),
                "score": float(score),
                "name": name
            })
        
        return results

    def save_facebank_append(self, new_embeddings: np.ndarray, new_names: list):
        current_embeddings, current_names = self.get_facebank()
        
        if current_embeddings.size > 0:
            all_embeddings = np.concatenate([current_embeddings, new_embeddings], axis=0)
            all_names = np.concatenate([current_names, np.array(new_names, dtype=object)], axis=0)
        else:
            all_embeddings = new_embeddings
            all_names = np.array(new_names, dtype=object)

        np.save(FACEBANK_EMBEDDINGS_DIR, all_embeddings)
        np.save(FACEBANK_NAMES_DIR, all_names)

        self.reset_facebank_cache() # Reload cache after saving