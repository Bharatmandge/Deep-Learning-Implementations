import torch 
import os 

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

#-- Datasets 
NUM_SYNTHETIC_IMAGES = 500 

IMAGE_SIZE = 256
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 2e-4
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_L1 = 100.0

GEN_FEATURES = 64
DISC_FEATURES = 64

IN_CHANNELS = 3
OUT_CHANNELS = 3

SAVE_IMAGE_EVERY = 5
SAVE_CHECKPOINT_EVERY = 10 
NUM_SAMPLES_TO_VISUALIZE = 4


SEED = 42

