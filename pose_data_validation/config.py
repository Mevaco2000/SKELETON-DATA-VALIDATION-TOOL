"""Configuration and constants for the pose_data_validation package."""

# Kept for backward compatibility (exported by the package); not used internally.
MODEL_REGISTRY = {
    "custom": None,
}

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_MODEL_PRETRAINED = "openai"

DEFAULT_K_NEIGHBORS = 10
DEFAULT_VAL_RATIO = 0.2
DEFAULT_RANDOM_SEED = 42
