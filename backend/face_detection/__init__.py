"""Face detection + clustering models used by the GPU worker pipeline.

In plain English: this package bundles three face-detection models (MTCNN,
YuNet, InsightFace), runs them in *consensus* (a face only counts if at
least two models agree), and clusters the resulting embeddings into one
group per person (DBSCAN over cosine distance).

Real package at ``backend/face_detection/`` — import normally.
"""

from face_detection.face_cluster import run_clustering
from face_detection.face_pipeline import FaceDetector, detect_hardware

__all__ = ["FaceDetector", "detect_hardware", "run_clustering"]
