from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "dataset"
TRAINING_DIR = BASE_DIR.parent / "training"

# File paths
training_images_path = DATASET_DIR / "training_set"
testing_images_path = DATASET_DIR / "testing_set"
res_im_path = TRAINING_DIR / "training_results"
models_path = TRAINING_DIR / "models"

# Training Parameters
low_resolution_shape = (64, 64, 3)
high_resolution_shape = (256, 256, 3)
batch_size = 2
epochs = 2000
n_epochs_per_image = 50

# Evaluation Parameters
model_to_be_evaluated = "trained_model.h5"
#n_images_to_be_evaluated = 100
