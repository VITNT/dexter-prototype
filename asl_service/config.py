from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SERVICE_ROOT.parent
OFFICIAL_I3D_ROOT = PROJECT_ROOT / "third_party" / "WLASL" / "code" / "I3D"
CLASS_LIST_PATH = OFFICIAL_I3D_ROOT / "preprocess" / "wlasl_class_list.txt"
CHECKPOINT_PATH = SERVICE_ROOT / "models" / "wlasl300.pt"

NUM_CLASSES = 300
FRAME_COUNT = 64
IMAGE_SIZE = 224

