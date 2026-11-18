"""Central configuration. All paths are relative to the repository root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
TRAIN_DIR = DATA_RAW / "train"
STUDENT_TEST_DIR = DATA_RAW / "student_test"
EVAL_DIR = DATA_RAW / "evaluator_test"          # placed on Laptop 1 only, never used in development
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MAPS = ROOT / "maps"
MODELS = ROOT / "models"

SEED = 42

# Candidate generation
CAND_RADIUS_M = 50.0      # search radius around each GPS point
CAND_MAX_K = 12           # keep at most K nearest arcs per point
SAMPLE_STEP_M = 10.0      # arc densification step used only for the spatial index

# Label generation (monotonic alignment of GPS points to the reference route)
LABEL_CLIP_M = 100.0      # distances clipped in the alignment cost (outlier robustness)
LABEL_RELIABLE_M = 50.0   # label considered reliable if point is within this distance of its arc

# Training subsample: cap points per trip so very long trips don't dominate
MAX_POINTS_PER_TRIP = 1500

# HMM gap handling
GAP_BREAK_S = 60.0        # time gap that resets the HMM chain
