"""Shared configuration for all ablation variants."""

# Paths
MODEL_PATH = '/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'
DATA_PATH = '/opt/onevl-experiment/data/navsim_real_traj_expanded.jsonl'
VIS_TOKENS_PATH = '/opt/onevl-experiment/data/emu3_visual_tokens_expanded.pt'
OUTPUT_BASE = '/opt/onevl-experiment/output/ablation'

# Training
MAX_SAMPLES = 1250
NUM_EVAL = 50  # Last 50 samples as held-out
GRAD_ACCUM = 8
LR = 2e-5
LORA_R = 64
LORA_ALPHA = 128

# Model architecture
CODEBOOK_SIZE = 32768
NUM_VIS_QUERIES = 64
NUM_VIS_LATENT = 4
NUM_LANG_LATENT = 2
VIS_LATENT_ID = 151662
LANG_LATENT_ID = 151663

# Loss weights
LAMBDA_VIS = 0.5
LAMBDA_LANG = 0.5

# Training schedule (OneVL recipe)
PRELIMINARY_EPOCHS = 3
STAGE_0_EPOCHS = 5
STAGE_1_EPOCHS = 3
STAGE_2_EPOCHS = 5

# Default CoT target (for language decoder)
DEFAULT_COT = "The vehicle should maintain safe trajectory based on the current driving conditions."
