from datasets import load_dataset
import numpy as np

# Load the dataset
dataset = load_dataset("ODELIA-AI/ODELIA-Challenge-2025", name="default")

# Access the training split 
ds_train = dataset['train']

# Retrieve a single sample from the training set
item = ds_train[0]

# Access an image
tensor = np.array(item['Image_T2'], dtype=np.int16)
affine = np.array(item['Affine_T2'], dtype=np.float64)

# Print metadata (excluding the image itself)
for key in item.keys():
    if not key.startswith('Image') and not key.startswith('Affine'):
        print(f"{key}: {item[key]}")