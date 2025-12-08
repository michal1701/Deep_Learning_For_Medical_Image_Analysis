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
num_images = len(all_image_paths)
val_count = int(0.10 * num_images)  # 10% for validation
# We take the last 10% of sorted data as validation for simplicity
train_image_paths = all_image_paths[:-val_count]
train_mask_paths  = all_mask_paths[:-val_count]
val_image_paths   = all_image_paths[-val_count:]
val_mask_paths    = all_mask_paths[-val_count:]

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
# # Task 2 - Data Augmentation and Improved Training Strategies

# %% [markdown]
#  ### Deterministic data augmentation (image + mask kept in sync)
#  Using stateless TF ops to guarantee identical random transforms for image & mask.

# %%
import tensorflow as tf

# Reuse: IMG_SIZE, BATCH_SIZE (from Week 1)
# We'll keep flips (safe for both image and mask). You can add more later (e.g., rotations via tfa).
def load_and_augment(image_path, mask_path):
    image, mask = load_image_and_mask(image_path, mask_path)
    # Use a shared random seed so both image and mask get identical transforms
    seed = tf.random.uniform(shape=[], maxval=10_000, dtype=tf.int32)
    # Horizontal flip
    image = tf.image.stateless_random_flip_left_right(image, seed=[seed, 0])
    mask  = tf.image.stateless_random_flip_left_right(mask,  seed=[seed, 0])
    # Vertical flip
    image = tf.image.stateless_random_flip_up_down(image,   seed=[seed, 1])
    mask  = tf.image.stateless_random_flip_up_down(mask,    seed=[seed, 1])
    # Optional: slight brightness jitter ONLY to image (never to mask)
    image = tf.image.stateless_random_brightness(image, max_delta=0.05, seed=[seed, 2])
    # Clip image back to [0,1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, mask

def make_dataset(image_paths, mask_paths, batch_size, augment=False, shuffle=True, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if augment:
        ds = ds.map(load_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(100, seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_aug_dataset = make_dataset(train_image_paths, train_mask_paths, BATCH_SIZE, augment=True, shuffle=True, seed=42)
val_dataset       = make_dataset(val_image_paths,   val_mask_paths,   BATCH_SIZE, augment=False, shuffle=False)

# %% [markdown]
# ### Metrics: IoU and Dice (for validation focus)

# %%
import tensorflow as tf

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    inter  = tf.reduce_sum(y_true * y_pred)
    union  = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter
    return (inter + smooth) / (union + smooth)

def dice_metric(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    inter  = tf.reduce_sum(y_true * y_pred)
    return (2.0 * inter + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# %% [markdown]
# ### Baseline (with augmentation) — same architecture, simple optimizer/loss
#  We keep the Week 1 U-Net, train on augmented data, and evaluate on clean validation.

# %%
# Reuse: get_unet_model from Week 1

baseline_aug_model = get_unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3))
baseline_aug_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # simple & strong baseline
    loss="binary_crossentropy",
    metrics=["accuracy", iou_metric, dice_metric]
)

early_stop_base = tf.keras.callbacks.EarlyStopping(
    monitor="val_iou_metric", mode="max", patience=5, restore_best_weights=True
)

history_baseline_aug = baseline_aug_model.fit(
    train_aug_dataset,
    epochs=15,
    validation_data=val_dataset,
    callbacks=[early_stop_base],
    verbose=1
)

base_eval = baseline_aug_model.evaluate(val_dataset, verbose=0)
print(f"[Baseline (aug)] Val -> Acc: {base_eval[1]:.3f}, IoU: {base_eval[2]:.3f}, Dice: {base_eval[3]:.3f}")

# %% [markdown]
# ### Hyperparameter search (small, fast, reliable)
# We try a few sane configs (optimizer + LR + loss) and pick the best on val IoU.

# %%
import copy
import numpy as np
from collections import OrderedDict

def dice_loss(y_true, y_pred, smooth=1.0):
    # differentiable dice loss
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice

def make_loss(loss_name="bce", alpha=0.7):
    if loss_name == "bce":
        return tf.keras.losses.BinaryCrossentropy()
    elif loss_name == "bce_dice":
        # Weighted combo: alpha*BCE + (1-alpha)*DiceLoss
        bce = tf.keras.losses.BinaryCrossentropy()
        def combo(y_true, y_pred):
            return alpha * bce(y_true, y_pred) + (1.0 - alpha) * dice_loss(y_true, y_pred)
        return combo
    else:
        raise ValueError("Unknown loss")

search_space = [
    # name, optimizer, lr, loss
    ("adam_1e-3_bce",      tf.keras.optimizers.Adam,      1e-3, "bce",      0.7),
    ("adam_5e-4_bce",      tf.keras.optimizers.Adam,      5e-4, "bce",      0.7),
    ("adam_1e-4_bce",      tf.keras.optimizers.Adam,      1e-4, "bce",      0.7),
    ("adam_1e-4_bce_dice", tf.keras.optimizers.Adam,      1e-4, "bce_dice", 0.7),
    ("rms_5e-4_bce",       tf.keras.optimizers.RMSprop,   5e-4, "bce",      0.7),
]

results = []
best_model = None
best_conf  = None
best_val_iou = -np.inf

for name, Opt, lr, loss_name, alpha in search_space:
    print(f"\n=== Training config: {name} ===")
    # fresh model each time
    model_t = get_unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3))
    model_t.compile(
        optimizer=Opt(learning_rate=lr),
        loss=make_loss(loss_name, alpha=alpha),
        metrics=["accuracy", iou_metric, dice_metric]
    )
    # early stopping on IoU
    cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_iou_metric", mode="max", patience=4, restore_best_weights=True
    )
    hist = model_t.fit(
        train_aug_dataset,
        epochs=20,
        validation_data=val_dataset,
        callbacks=[cb],
        verbose=1
    )
    eval_vals = model_t.evaluate(val_dataset, verbose=0)
    res = OrderedDict(
        name=name,
        lr=lr,
        loss=loss_name,
        acc=float(eval_vals[1]),
        iou=float(eval_vals[2]),
        dice=float(eval_vals[3])
    )
    results.append(res)
    print(f"[{name}] Val -> Acc: {res['acc']:.3f}, IoU: {res['iou']:.3f}, Dice: {res['dice']:.3f}")
    if res["iou"] > best_val_iou:
        best_val_iou = res["iou"]
        best_model = model_t  # keep weights restored by early stopping
        best_conf  = res

print("\n=== Search summary ===")
for r in results:
    print(f"{r['name']}: IoU={r['iou']:.3f}, Dice={r['dice']:.3f}, Acc={r['acc']:.3f}")

print("\nBest config by IoU:")
print(best_conf)

# %% [markdown]
# ### 2.5 Evaluate best model & quick visual check

# %%
# Evaluate on validation set (again, for clarity)
best_eval = best_model.evaluate(val_dataset, verbose=0)
print(f"[Best Model] Val -> Acc: {best_eval[1]:.3f}, IoU: {best_eval[2]:.3f}, Dice: {best_eval[3]:.3f}")

# %%
# Visualize a few predictions from the best model
import numpy as np
import matplotlib.pyplot as plt

sample_images, sample_masks = next(iter(val_dataset))
preds_best = (best_model.predict(sample_images) > 0.5).astype("float32")

n_show = min(3, sample_images.shape[0])
plt.figure(figsize=(12, 4*n_show))
for i in range(n_show):
    plt.subplot(n_show, 3, 3*i + 1)
    plt.imshow(sample_images[i])
    plt.title("Input")
    plt.axis("off")
    plt.subplot(n_show, 3, 3*i + 2)
    plt.imshow(sample_masks[i, ..., 0], cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")
    plt.subplot(n_show, 3, 3*i + 3)
    plt.imshow(preds_best[i, ..., 0], cmap="gray")
    plt.title("Best Model Prediction")
    plt.axis("off")
plt.tight_layout()
plt.show()

# %%



