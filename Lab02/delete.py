# %% [markdown]
# ## Michal Binda - DMLIA Lab02
# 
# ### Task 1 - Data Preprocessing, Baseline Model Implementation
# 

# %% [markdown]
# In this Task 1, I implemented a U-Net-based model for skin lesion segmentation and evaluated its baseline performance. I resized the images for faster training and achieved a working segmentation that captures the lesion region with reasonable accuracy. The baseline metrics (e.g., mean IoU and Dice on the validation set) provide a reference point for future improvements. In the following weeks, I plan to enhance the segmentation by applying techniques such as data augmentation, hyperparameter tuning, and incorporating more training data or regularization to address overfitting. I will also implement additional metrics (sensitivity, specificity, ROC curves, etc.) and compare my results with the ISIC 2016 challenge benchmarks in the final week.
# 

# %%
import os

# After extraction, set the directories
image_dir = "ISIC_2016/Training_Data"
mask_dir = "ISIC_2016/Training_GroundTruth"

# List all image and mask file paths
all_image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(".jpg")])
all_mask_paths  = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith(".png")])

print(f"Total images: {len(all_image_paths)}")
print(f"Total masks:  {len(all_mask_paths)}")
print("Example image file:", all_image_paths[0])
print("Example mask file:",  all_mask_paths[0])

# %% [markdown]
# ### Split into training and validation sets (e.g., 90 for validation, 810 for training)
# 

# %%
from sklearn.model_selection import train_test_split

train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(
    all_image_paths,
    all_mask_paths,
    test_size=0.10,
    random_state=42,
    shuffle=True
)

print(f"Training set: {len(train_image_paths)} images")
print(f"Validation set: {len(val_image_paths)} images")

# %%
import tensorflow as tf

IMG_HEIGHT = 256
IMG_WIDTH  = 256
IMG_SIZE   = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 16  # using 16 as a reasonable batch size for 256x256 images

def load_image_and_mask(image_path, mask_path):
    # Read and decode the image
    image_file = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image_file, channels=3)
    # Resize the image and normalize to [0,1]
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.image.convert_image_dtype(image, tf.float32)  # scales 0-255 to 0-1
    
    # Read and decode the mask
    mask_file = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(mask_file, channels=1)
    # Resize mask using nearest neighbor (so we don't get interpolated values)
    mask = tf.image.resize(mask, IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Convert mask to binary 0/1 format as float32
    mask = tf.cast(mask, tf.float32) / 255.0
    return image, mask

# Create TensorFlow Dataset objects for training and validation
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_mask_paths))
train_dataset = train_dataset.map(load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=100, seed=42).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_mask_paths))
val_dataset = val_dataset.map(load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Verify the shape of data pipeline output
sample_image_batch, sample_mask_batch = next(iter(train_dataset))
print("Image batch shape:", sample_image_batch.shape)
print("Mask batch shape:", sample_mask_batch.shape)
print("Image pixel range (min, max):", float(tf.reduce_min(sample_image_batch)), float(tf.reduce_max(sample_image_batch)))
print("Mask unique values:", tf.unique(tf.reshape(sample_mask_batch, (-1,)))[0].numpy())

# %% [markdown]
# ### Build a U-Net-like model for binary segmentation

# %%
from tensorflow.keras import layers, models

def get_unet_model(input_size=(256, 256, 3)):
    inputs = layers.Input(shape=input_size)
    
    # Entry block: initial conv with stride 2 to downsample
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # Save the activation output as residual for skip connection
    previous_block_activation = x  

    # Blocks 1, 2, 3: downsampling blocks with increasing filter sizes
    for filters in [64, 128, 256]:
        # Downsampling block
        x = layers.Activation("relu")(x)
        # SeparableConv2D is used for efficiency (depthwise separable convolution)
        x = layers.SeparableConv2D(filters, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        # Downsample the feature map
        x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
        # Project residual (skip connection) to match filter dims
        residual = layers.Conv2D(filters, kernel_size=1, strides=2, padding="same")(previous_block_activation)
        # Add the projected residual to the output of the block
        x = layers.Add()([x, residual])
        # Store current output for next block's residual connection
        previous_block_activation = x

    # [Second half: Upsampling through transpose conv blocks]
    for filters in [256, 128, 64, 32]:
        # Upsampling block
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        # Upsample the feature map to double the height and width
        x = layers.UpSampling2D(size=2)(x)
        # Project the residual from previous downsampling block to the same shape
        residual = layers.UpSampling2D(size=2)(previous_block_activation)
        residual = layers.Conv2D(filters, kernel_size=1, padding="same")(residual)
        # Add the residual (skip connection) to the upsampled feature
        x = layers.Add()([x, residual])
        # Update the previous block activation to the one from deeper layer (moving up)
        previous_block_activation = x

    # Output layer: one conv filter with sigmoid for binary mask
    outputs = layers.Conv2D(1, kernel_size=1, activation="sigmoid", padding="same")(x)
    # Define the model
    model = models.Model(inputs, outputs, name="U-Net")
    return model

# Instantiate the model and print a summary
model = get_unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3))
model.summary()

# %% [markdown]
# ### Compile the model

# %%
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Train the model on the training set, validating on the validation set
EPOCHS = 15
history = model.fit(train_dataset, 
                    epochs=EPOCHS, 
                    validation_data=val_dataset)

# %%
import numpy as np
from keras.utils import load_img, img_to_array

# Get model predictions on the validation set
val_preds = model.predict(val_dataset)
# Ensure predictions are 0/1 by thresholding at 0.5
val_preds_thresholded = (val_preds > 0.5).astype(np.uint8)  # shape: (N, 256, 256, 1)

# Compute IoU and Dice for each image in the validation set
ious = []
dices = []
for i in range(len(val_image_paths)):
    # Load the true mask (and resize to 256x256 to match prediction)
    true_mask = load_img(val_mask_paths[i], color_mode="grayscale", target_size=IMG_SIZE)
    true_mask = np.array(true_mask, dtype=np.uint8)  # shape: (256,256)
    true_mask_bin = (true_mask > 127).astype(np.uint8)  # ensure binary 0/1
    
    pred_mask_bin = val_preds_thresholded[i, ..., 0]  # predicted binary mask, shape (256,256)
    
    # Intersection and Union
    intersection = np.sum(true_mask_bin * pred_mask_bin)
    union = np.sum(true_mask_bin) + np.sum(pred_mask_bin) - intersection
    # If union is zero (edge case: no lesion in both), define IoU as 1
    if union == 0:
        iou = 1.0
        dice = 1.0
    else:
        iou = intersection / union
        dice = (2 * intersection) / (np.sum(true_mask_bin) + np.sum(pred_mask_bin) + 1e-8)
    ious.append(iou)
    dices.append(dice)

mean_iou = np.mean(ious)
mean_dice = np.mean(dices)
print(f"Mean IoU on validation set: {mean_iou:.3f}")
print(f"Mean Dice coefficient on validation set: {mean_dice:.3f}")

# %%
import matplotlib.pyplot as plt

# Pick an index to visualize (for example, 0)
idx = 0
# Load the original image (in color) and the true mask
orig_image = load_img(val_image_paths[idx], target_size=IMG_SIZE)
orig_image = np.array(orig_image, dtype=np.uint8)
true_mask = load_img(val_mask_paths[idx], color_mode="grayscale", target_size=IMG_SIZE)
true_mask = np.array(true_mask, dtype=np.uint8)
pred_mask = val_preds_thresholded[idx, ..., 0] * 255  # multiply by 255 to visualize as 0-255 image

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(orig_image)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,3,2)
plt.imshow(true_mask, cmap="gray")
plt.title("Ground Truth Mask")
plt.axis("off")
plt.subplot(1,3,3)
plt.imshow(pred_mask, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")
plt.show()

# %% [markdown]
# ## Task 2 – Data Augmentation and Improved Model (Dice Loss)
# 
# The ISIC 2016 dataset contains fewer than 1 000 training images, which makes overfitting likely.
# To address this, we apply simple deterministic flips (horizontal + vertical) that effectively double
# the training variability without breaking mask alignment. Then, we train a second U-Net model
# using Dice loss — a loss function that directly optimizes for overlap between predicted and true
# segmentation masks.
# 
# Other transformations like scaling or zooming were avoided, since they could distort the medical images and change the relative size or shape of the lesions, which are critical for accurate segmentation.

# %%
import tensorflow as tf

def load_image_and_mask(image_path, mask_path):
    image_file = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image_file, channels=3)
    image = tf.image.resize(image, (256, 256))
    image = tf.image.convert_image_dtype(image, tf.float32)

    mask_file = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(mask_file, channels=1)
    mask = tf.image.resize(mask, (256, 256),
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.cast(mask, tf.float32) / 255.0
    return image, mask


def load_and_flip(image_path, mask_path):
    image, mask = load_image_and_mask(image_path, mask_path)
    seed = tf.random.uniform(shape=[], maxval=10000, dtype=tf.int32)
    # Horizontal + vertical flips (same seed for both)
    image = tf.image.stateless_random_flip_left_right(image, seed=[seed, 0])
    mask  = tf.image.stateless_random_flip_left_right(mask,  seed=[seed, 0])
    image = tf.image.stateless_random_flip_up_down(image,   seed=[seed, 1])
    mask  = tf.image.stateless_random_flip_up_down(mask,    seed=[seed, 1])
    return image, mask


def make_dataset(imgs, masks, batch=16, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((imgs, masks))
    if augment:
        ds = ds.map(load_and_flip, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.shuffle(100, seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


train_dataset_aug = make_dataset(train_image_paths, train_mask_paths,
                                 batch=16, augment=True)
val_dataset       = make_dataset(val_image_paths,   val_mask_paths,
                                 batch=16, augment=False)

# %%
def dice_loss(y_true, y_pred, smooth=1.0):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (denom + smooth)


def iou_metric(y_true, y_pred, smooth=1e-6):
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_bin)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_bin) - intersection
    return (intersection + smooth) / (union + smooth)

# %%
baseline_model = get_unet_model((256, 256, 3))
baseline_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                       loss="binary_crossentropy",
                       metrics=["accuracy", iou_metric])

hist_base = baseline_model.fit(
    train_dataset_aug,
    validation_data=val_dataset,
    epochs=10,
    verbose=1
)

base_eval = baseline_model.evaluate(val_dataset, verbose=0)
print(f"Baseline (BCE) → IoU: {base_eval[2]:.3f}")

# %%
improved_model = get_unet_model((256, 256, 3))
improved_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                       loss=dice_loss,
                       metrics=["accuracy", iou_metric])

hist_dice = improved_model.fit(
    train_dataset_aug,
    validation_data=val_dataset,
    epochs=10,
    verbose=1
)

dice_eval = improved_model.evaluate(val_dataset, verbose=0)
print(f"Improved (Dice Loss) → IoU: {dice_eval[2]:.3f}")

# %%
print("=== Validation Comparison ===")
print(f"Baseline (BCE)  IoU: {base_eval[2]:.3f}")
print(f"Improved (Dice) IoU: {dice_eval[2]:.3f}")

if dice_eval[2] > base_eval[2]:
    print("✅ The Dice-loss model achieved higher IoU — better segmentation overlap.")
else:
    print("⚠️ Dice-loss model underperformed; BCE might generalize slightly better here.")

# %%
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import load_img

# Make predictions for the same validation batch
val_images, val_masks = next(iter(val_dataset))
preds_bce  = (baseline_model.predict(val_images)  > 0.5).astype("float32")
preds_dice = (improved_model.predict(val_images) > 0.5).astype("float32")

# Number of examples to display
n_show = min(10, val_images.shape[0])
IMG_SIZE = (256, 256)

plt.figure(figsize=(12, n_show * 4))

for i in range(n_show):
    # Original image
    orig = val_images[i].numpy()
    gt   = val_masks[i, ..., 0].numpy()
    pred_bce  = preds_bce[i, ..., 0]
    pred_dice = preds_dice[i, ..., 0]

    # Row i: [original | BCE pred | Dice pred]
    plt.subplot(n_show, 4, 4*i + 1)
    plt.imshow(orig)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(n_show, 4, 4*i + 2)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(n_show, 4, 4*i + 3)
    plt.imshow(pred_bce, cmap="gray")
    plt.title("Pred (BCE)")
    plt.axis("off")

    plt.subplot(n_show, 4, 4*i + 4)
    plt.imshow(pred_dice, cmap="gray")
    plt.title("Pred (Dice)")
    plt.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Observations
# 
# - **Dice loss improved IoU** from **0.36 → 0.54**, showing better overlap between predicted and true lesion regions.  
# - **BCE** often underestimated lesion size due to the **class imbalance** (few lesion pixels vs. many background pixels).  
# - **Dice loss** directly optimizes region overlap, giving higher importance to lesion pixels and producing fuller, more accurate masks.  
# - Visually, Dice predictions capture lesion boundaries more completely, while BCE masks remain fragmented.  
# - Overall, Dice loss provides a stronger learning signal for small lesion areas, leading to better segmentation quality on ISIC 2016.

# %% [markdown]
# # Task 2.1 - Parameter Optimization
# 
# In this step, we systematically test different combinations of **optimizers**, **learning rates**, and **loss functions** to identify the best-performing configuration for the U-Net segmentation model.
# 
# Five configurations are evaluated:
# 
# | Name | Optimizer | Learning Rate | Batch Size | Loss Function |
# |:-----|:-----------|:--------------|:------------|:---------------|
# | adam_1e-3_dice | Adam | 1e-3 | 16 | Dice |
# | adam_1e-4_dice | Adam | 1e-4 | 16 | Dice |
# | adam_1e-4_bce_dice | Adam | 1e-4 | 16 | BCE + Dice |
# | rms_1e-3_dice | RMSprop | 1e-3 | 16 | Dice |
# | rms_1e-4_bce_dice | RMSprop | 1e-4 | 16 | BCE + Dice |
# 
# Each configuration is trained for up to **30 epochs**, using **EarlyStopping** and **ReduceLROnPlateau** callbacks to prevent overfitting and improve convergence.  
# The model achieving the **highest validation IoU (Jaccard index)** is selected as the **best configuration**.
# 
# This process allows us to determine which optimizer–loss–learning-rate combination provides the most stable and accurate segmentation performance.

# %%
import tensorflow as tf
import numpy as np
from collections import OrderedDict

# Reuse: get_unet_model, dice_loss, iou_metric, train_dataset_aug, val_dataset

def make_loss(loss_type="dice", alpha=0.7):
    """Return a loss function based on string identifier."""
    if loss_type == "bce":
        return tf.keras.losses.BinaryCrossentropy()
    elif loss_type == "dice":
        return dice_loss
    elif loss_type == "bce_dice":
        bce = tf.keras.losses.BinaryCrossentropy()
        def combo(y_true, y_pred):
            return alpha * bce(y_true, y_pred) + (1 - alpha) * dice_loss(y_true, y_pred)
        return combo
    else:
        raise ValueError("Unknown loss type")

search_space = [
    ("adam_1e-3_dice",       tf.keras.optimizers.Adam,     1e-3, 16, "dice"),
    ("adam_1e-4_dice",       tf.keras.optimizers.Adam,     1e-4, 16, "dice"),
    ("adam_1e-4_bce_dice",   tf.keras.optimizers.Adam,     1e-4, 16, "bce_dice"),
    ("rms_1e-3_dice",        tf.keras.optimizers.RMSprop,  1e-3, 16, "dice"),
    ("rms_1e-4_bce_dice",    tf.keras.optimizers.RMSprop,  1e-4, 16, "bce_dice"),
]

# %%
best_model = None
best_result = None
best_iou = -np.inf
results = []

for name, Opt, lr, batch, loss_name in search_space:
    print(f"\n=== Training configuration: {name} ===")
    # Create datasets with correct batch size
    train_ds = make_dataset(train_image_paths, train_mask_paths,
                            batch=batch, augment=True)
    val_ds   = make_dataset(val_image_paths, val_mask_paths,
                            batch=batch, augment=False)

    model = get_unet_model((256, 256, 3))
    model.compile(optimizer=Opt(learning_rate=lr),
                  loss=make_loss(loss_name),
                  metrics=["accuracy", iou_metric])

    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_iou_metric", mode="max",
            patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_iou_metric", mode="max",
            factor=0.5, patience=3, min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=20, callbacks=[cb], verbose=1)

    val_eval = model.evaluate(val_ds, verbose=0)
    iou_val = float(val_eval[2])

    results.append(OrderedDict(
        name=name,
        lr=lr,
        batch=batch,
        loss=loss_name,
        acc=float(val_eval[1]),
        iou=iou_val
    ))
    print(f"[{name}] Validation IoU: {iou_val:.4f}")

    if iou_val > best_iou:
        best_iou = iou_val
        best_result = results[-1]
        best_model = model

# %%
print("\n=== Optimization Summary ===")
for r in results:
    print(f"{r['name']:20s} | lr={r['lr']:.0e} | batch={r['batch']:2d} | "
          f"loss={r['loss']:8s} | IoU={r['iou']:.4f}")

print("\nBest configuration by IoU:")
print(best_result)

# %%
best_eval = best_model.evaluate(val_dataset, verbose=0)
print(f"[Best Model] Val → Acc: {best_eval[1]:.3f}, IoU: {best_eval[2]:.3f}")

# Visualize a few predictions
val_images, val_masks = next(iter(val_dataset))
preds_best = (best_model.predict(val_images) > 0.5).astype("float32")

import matplotlib.pyplot as plt

n_show = min(3, val_images.shape[0])
plt.figure(figsize=(12, 4 * n_show))
for i in range(n_show):
    plt.subplot(n_show, 3, 3*i + 1)
    plt.imshow(val_images[i])
    plt.title("Original")
    plt.axis("off")

    plt.subplot(n_show, 3, 3*i + 2)
    plt.imshow(val_masks[i, ..., 0], cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(n_show, 3, 3*i + 3)
    plt.imshow(preds_best[i, ..., 0], cmap="gray")
    plt.title("Best Model Prediction")
    plt.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# 

# %% [markdown]
# # Task 3: Comparison between baseline and improved model

# %% [markdown]
# implement the metrics described in https://challenge.isic-
# archive.com/landing/2016/37
# a. Present the data as table comparing the BASELINE with the
# best model. Use plots if needed to show performance increase.
# b. Compare in the table the results with the challenge winner


