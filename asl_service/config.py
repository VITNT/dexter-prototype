from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parent
CLASS_LIST_PATH = SERVICE_ROOT / "models" / "wlasl2000_labels.txt"
CHECKPOINT_PATH = SERVICE_ROOT / "models" / "openhands_wlasl2000_stgcn.ckpt"

MODEL_NAME = "OpenHands ST-GCN"
NUM_CLASSES = 2000
INPUT_LANDMARK_COUNT = 75
INPUT_COORDINATE_COUNT = 3
INPUT_FEATURE_COUNT = INPUT_LANDMARK_COUNT * INPUT_COORDINATE_COUNT
MODEL_LANDMARK_COUNT = 27
MODEL_COORDINATE_COUNT = 2

# WLASL2000 has no idle/no-sign class. These gates prevent the closed-set model
# from turning an empty or ambiguous landmark window into an arbitrary word.
MIN_MOTION_SCORE = 0.0025
MIN_TOP1_CONFIDENCE = 0.20
MIN_TOP1_MARGIN = 0.02
