import os
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SERVICE_ROOT.parent
RUNNING_ON_VERCEL = bool(os.getenv("VERCEL"))
configured_runtime_root = os.getenv("WLASL_RUNTIME_ROOT", "").strip()
RUNTIME_ROOT = (
    Path(configured_runtime_root).expanduser()
    if configured_runtime_root
    else Path("/tmp/dexter-wlasl")
    if RUNNING_ON_VERCEL
    else PROJECT_ROOT
)
OFFICIAL_I3D_ROOT = RUNTIME_ROOT / "third_party" / "WLASL" / "code" / "I3D"
CLASS_LIST_PATH = OFFICIAL_I3D_ROOT / "preprocess" / "wlasl_class_list.txt"
CHECKPOINT_PATH = RUNTIME_ROOT / "asl_service" / "models" / "wlasl300.pt"

NUM_CLASSES = 300
FRAME_COUNT = 64
IMAGE_SIZE = 224
RESIZE_SIZE = 256
default_upload_limit = 4 * 1024 * 1024 if RUNNING_ON_VERCEL else 20 * 1024 * 1024
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(default_upload_limit)))

# WLASL has no idle/no-sign class. These gates prevent the closed-set model
# from turning an empty or ambiguous webcam window into an arbitrary word.
MIN_MOTION_SCORE = 0.0025
MIN_TOP1_CONFIDENCE = 0.20
MIN_TOP1_MARGIN = 0.02
