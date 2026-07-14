from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SERVICE_ROOT.parent
OFFICIAL_I3D_ROOT = PROJECT_ROOT / "third_party" / "WLASL" / "code" / "I3D"
CLASS_LIST_PATH = OFFICIAL_I3D_ROOT / "preprocess" / "wlasl_class_list.txt"
CHECKPOINT_PATH = SERVICE_ROOT / "models" / "wlasl300.pt"

NUM_CLASSES = 300
FRAME_COUNT = 64
IMAGE_SIZE = 224
RESIZE_SIZE = 256

# WLASL has no idle/no-sign class. These gates prevent the closed-set model
# from turning an empty or ambiguous webcam window into an arbitrary word.
MIN_MOTION_SCORE = 0.0025
MIN_TOP1_CONFIDENCE = 0.20
MIN_TOP1_MARGIN = 0.02
