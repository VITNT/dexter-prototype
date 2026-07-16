from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parent
CLASS_LIST_PATH = SERVICE_ROOT / "models" / "wlasl300_labels.txt"
CHECKPOINT_PATH = SERVICE_ROOT / "models" / "phonssm_wlasl300.pt"

MODEL_NAME = "PhonSSM"
NUM_CLASSES = 300
FRAME_COUNT = 30
LANDMARK_COUNT = 75
COORDINATE_COUNT = 3
FEATURE_COUNT = LANDMARK_COUNT * COORDINATE_COUNT

# The WLASL300 PhonSSM benchmark uses pose + left hand + right hand and the
# adaptive temperature selected by the reference training script.
INPUT_MODE = "pose_hands"
TEMPERATURE = 0.5

# WLASL has no idle/no-sign class. These gates prevent the closed-set model
# from turning an empty or ambiguous landmark window into an arbitrary word.
MIN_MOTION_SCORE = 0.0025
MIN_TOP1_CONFIDENCE = 0.20
MIN_TOP1_MARGIN = 0.02
