import os
import math
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer, Lambda, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Conv2D

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("TPU detected and initialized")
    print(f"Number of accelerators: {strategy.num_replicas_in_sync}")
except Exception as e:
    print("TPU not found or failed to initialize, falling back to default strategy.")
    strategy = tf.distribute.get_strategy()
    print(f"Using strategy: {type(strategy).__name__}")

SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------------------------------------------------
# Configurations
# -------------------------------------------------------------------
TEST_CONFIG = {
    'num_samples': 300,      # total across persons
    'image_size': 256,
    'batch_size': 16,
    'epochs': 5,
    'learning_rate': 1e-4,
    'min_learning_rate': 1e-6,
    'warmup_epochs': 2
}

TEST_FINETUNE_CONFIG = {
    'num_samples': 300,
    'batch_size': 16,
    'epochs': 3,
    'learning_rate': 1e-4,
    'min_learning_rate': 1e-7
}

FULL_CONFIG = {
    'num_samples': 60000,    # total across persons
    'image_size': 256,
    'batch_size': 16 * max(1, strategy.num_replicas_in_sync // 2),  
    'epochs': 170,
    'learning_rate': 1e-4,
    'min_learning_rate': 1e-6,
    'warmup_epochs': 5
}

FULL_FINETUNE_CONFIG = {
    'num_samples': 60000,
    'batch_size': 16 * max(1, strategy.num_replicas_in_sync // 2),
    'epochs': 90,
    'learning_rate': 1e-4,
    'min_learning_rate': 1e-7
}

# Select which to run
CONFIG = FULL_CONFIG
FINETUNE_CONFIG = FULL_FINETUNE_CONFIG

# Edge sampling settings
EDGE_GRID_SIZE = 3        # 3x3 grid
EDGE_TARGET_RATIO = 0.70  # In edge-training, target composition: 70% edge, 30% center

# -------------------------------------------------------------------
# Preprocessing and augmentation
# -------------------------------------------------------------------
# EfficientNetV2 preprocessing
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L, preprocess_input

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.GaussianNoise(0.01),
    tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1),
                               width_factor=(-0.1, 0.1),
                               fill_mode='constant')
], name="data_augmentation")

# -------------------------------------------------------------------
# Custom Layers
# -------------------------------------------------------------------
class GroupedCoordinateAttention(Layer):
    def __init__(self, reduction_ratio=16, groups=4, **kwargs):
        super(GroupedCoordinateAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.groups = groups

    def build(self, input_shape):
        channel = input_shape[-1]
        reduced_channels = max(channel // (self.reduction_ratio * self.groups), 8)

        self.conv_groups = []
        for _ in range(self.groups):
            group_convs = {
                'conv_h': Conv2D(reduced_channels, 1, use_bias=False, kernel_initializer='he_normal'),
                'conv_w': Conv2D(reduced_channels, 1, use_bias=False, kernel_initializer='he_normal'),
                'bn_h': BatchNormalization(),
                'bn_w': BatchNormalization()
            }
            self.conv_groups.append(group_convs)

        self.spatial_conv = Conv2D(self.groups, 7, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.spatial_bn = BatchNormalization()

        channel_per_group = channel // self.groups
        self.final_convs = [
            Conv2D(channel_per_group, 1, use_bias=False, kernel_initializer='he_normal')
            for _ in range(self.groups)
        ]

    def call(self, inputs):
        channel = inputs.shape[-1]
        channel_per_group = channel // self.groups

        grouped_inputs = tf.split(inputs, self.groups, axis=-1)
        outputs = []

        for idx, group_input in enumerate(grouped_inputs):
            x_h = Lambda(lambda x: tf.reduce_mean(x, axis=2, keepdims=True))(group_input)
            x_w = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(group_input)

            conv_group = self.conv_groups[idx]
            x_h = conv_group['conv_h'](x_h)
            x_w = conv_group['conv_w'](x_w)
            x_h = conv_group['bn_h'](x_h)
            x_w = conv_group['bn_w'](x_w)
            x_h = tf.nn.relu(x_h)
            x_w = tf.nn.relu(x_w)

            attention = tf.sigmoid(self.final_convs[idx](x_h + x_w))
            outputs.append(group_input * attention)

        spatial = self.spatial_conv(inputs)
        spatial = self.spatial_bn(spatial)
        spatial = tf.nn.sigmoid(spatial)
        spatial = tf.split(spatial, self.groups, axis=-1)

        outputs = [out * sp for out, sp in zip(outputs, spatial)]
        return tf.concat(outputs, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio, "groups": self.groups})
        return config

class PreActResidualBlock(Layer):
    def __init__(self, units, dropout_rate=0.3, **kwargs):
        super(PreActResidualBlock, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.bn1 = BatchNormalization()
        self.dense1 = Dense(units, kernel_initializer='he_normal')
        self.dropout = Dropout(dropout_rate)
        self.bn2 = BatchNormalization()
        self.dense2 = Dense(units, kernel_initializer='he_normal')
        self.shortcut = None

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.shortcut = Dense(self.units, kernel_initializer='he_normal')

    def call(self, inputs, training=None):
        x = self.bn1(inputs, training=training)
        x = tf.nn.silu(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.bn2(x, training=training)
        x = tf.nn.silu(x)
        x = self.dense2(x)
        shortcut = self.shortcut(inputs) if self.shortcut is not None else inputs
        return x + shortcut

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "dropout_rate": self.dropout_rate})
        return config

# -------------------------------------------------------------------
# Model builder
# -------------------------------------------------------------------
def build_enhanced_model(input_shape=(256, 256, 3), backbone='efficientnetv2l', trainable_tail=40):
    input_layer = keras.Input(shape=input_shape)
    if backbone == 'efficientnetv2l':
        backbone_model = keras.applications.efficientnet_v2.EfficientNetV2L(
            include_top=False, weights='imagenet', input_tensor=input_layer
        )
    # Freeze all but the last 'trainable_tail' layers
    for layer in backbone_model.layers[:-trainable_tail]:
        layer.trainable = False

    x = backbone_model.output
    x = GroupedCoordinateAttention(reduction_ratio=16, groups=4)(x)
    x = GlobalAveragePooling2D()(x)

    x = PreActResidualBlock(512, dropout_rate=0.4)(x)
    x = PreActResidualBlock(256, dropout_rate=0.3)(x)
    x = PreActResidualBlock(128, dropout_rate=0.2)(x)

    outputs = Dense(2, activation='sigmoid', kernel_initializer='glorot_normal')(x)
    return keras.Model(input_layer, outputs, name="efficientnetv2l_gca_regressor")

# -------------------------------------------------------------------
# Loss
# -------------------------------------------------------------------
def combined_loss(y_true, y_pred):
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    huber = tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
    combined = 0.7 * tf.reduce_mean(huber) + 0.3 * tf.reduce_mean(mse)
    return tf.where(tf.math.is_nan(combined), 1e3, combined)

# -------------------------------------------------------------------
# Edge/Center grid helpers (3x3)
# -------------------------------------------------------------------
def compute_grid_indices(norm_xy, grid_size=3):
    """
    norm_xy: (N,2) normalized x,y in [0,1]
    Returns two arrays x_idx, y_idx in {0,...,grid_size-1}
    """
    x = np.clip(norm_xy[:, 0], 0.0, 1.0 - 1e-8)
    y = np.clip(norm_xy[:, 1], 0.0, 1.0 - 1e-8)

    edges = np.linspace(0, 1, grid_size + 1)
    x_idx = np.digitize(x, edges) - 1
    y_idx = np.digitize(y, edges) - 1
    return x_idx, y_idx

def get_edge_center_masks(norm_xy, grid_size=3):
    """
    Defines the center as the middle cell only (for 3x3: cell (1,1)),
    and edge as all other cells.
    Returns boolean masks (edge_mask, center_mask)
    """
    assert grid_size == 3, "This implementation assumes a 3x3 grid."
    x_idx, y_idx = compute_grid_indices(norm_xy, grid_size)
    mid = grid_size // 2  # 1 for 3x3
    center_mask = (x_idx == mid) & (y_idx == mid)
    edge_mask = ~center_mask
    return edge_mask, center_mask

def rebalance_edge_center(X, y, edge_ratio=0.7, grid_size=3, seed=SEED):
    """
    Rebalances (with replacement) to reach the desired edge_ratio of total size.
    Returns X_balanced, y_balanced, and masks for info.
    """
    rng = np.random.default_rng(seed)
    edge_mask, center_mask = get_edge_center_masks(y, grid_size=grid_size)

    edge_idx = np.where(edge_mask)[0]
    center_idx = np.where(center_mask)[0]

    total = len(y)
    n_edge_target = int(total * edge_ratio)
    n_center_target = total - n_edge_target

    # If one pool is very small, sampling with replacement will duplicate entries as needed
    chosen_edge = rng.choice(edge_idx, size=n_edge_target, replace=True) if len(edge_idx) > 0 else np.array([], dtype=int)
    chosen_center = rng.choice(center_idx, size=n_center_target, replace=True) if len(center_idx) > 0 else np.array([], dtype=int)

    chosen = np.concatenate([chosen_edge, chosen_center])
    rng.shuffle(chosen)

    return X[chosen], y[chosen], edge_mask, center_mask

# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
def load_and_prepare_multi_person_data(csv_paths, video_paths, num_samples, image_size, edge_focused=False, edge_ratio=EDGE_TARGET_RATIO):
    """
    Reads frames and coordinates for multiple people/sources.
    - csv_paths: list of CSV paths [frame_index, x, y]
    - video_paths: matching list of video paths
    - num_samples: total desired samples across all persons (will be split evenly)
    - image_size: target square size (H=W=image_size)
    - edge_focused: if True, rebalance to increase edge samples using 3x3 definition
    - edge_ratio: target fraction of edge samples in edge_focused mode
    Returns: X (N,H,W,3), y (N,2 normalized), (orig_width, orig_height)
    """
    assert len(csv_paths) == len(video_paths), "csv_paths and video_paths must have same length"
    print("Loading and preparing multi-person data...")

    all_X, all_y = [], []
    orig_dimensions = None

    samples_per_person = num_samples // len(csv_paths) if num_samples else None

    for csv_path, video_path in zip(csv_paths, video_paths):
        print(f"  - Reading: {csv_path}")
        df = pd.read_csv(csv_path)
        frame_indices = df.iloc[:, 0].values
        coordinates = df.iloc[:, 1:3].values  # pixel coords in original video

        # Subsample per person if requested
        if samples_per_person and samples_per_person < len(frame_indices):
            sel = np.random.choice(len(frame_indices), samples_per_person, replace=False)
            frame_indices = frame_indices[sel]
            coordinates = coordinates[sel]

        # Open video and figure out dimensions
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if orig_dimensions is None:
            orig_dimensions = (width, height)
        else:
            # If different videos have different sizes, we still normalize by their own size
            # but consistency in evaluation should use matched video per CSV when evaluating.
            pass

        # Pre-allocate
        X = np.zeros((len(frame_indices), image_size, image_size, 3), dtype=np.float32)

        # Extract frames
        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                # If the frame can't be read, skip by repeating a previous valid sample if available
                if i > 0:
                    X[i] = X[i-1]
                else:
                    X[i] = 0.0
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            X[i] = frame
            if (i + 1) % 1000 == 0:
                print(f"    Processed {i + 1}/{len(frame_indices)} frames")

        cap.release()

        # EfficientNet preprocess in chunks (to reduce peak memory)
        chunk = 1000
        for s in range(0, len(X), chunk):
            e = min(s + chunk, len(X))
            X[s:e] = preprocess_input(X[s:e])

        # Normalize labels to [0,1] by original video size
        y = coordinates.astype(np.float32) / np.array([width, height], dtype=np.float32)
        y = np.clip(y, 1e-7, 1.0 - 1e-7)

        all_X.append(X)
        all_y.append(y)

    # Combine all persons
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    # Shuffle once
    idx = np.random.permutation(len(X_combined))
    X_combined, y_combined = X_combined[idx], y_combined[idx]

    # Apply edge-focused rebalance if requested (3x3 grid; center = middle cell)
    if edge_focused:
        print(f"Applying edge-focused rebalance: target edge ratio = {edge_ratio:.2f}")
        X_combined, y_combined, edge_mask, center_mask = rebalance_edge_center(
            X_combined, y_combined, edge_ratio=edge_ratio, grid_size=EDGE_GRID_SIZE, seed=SEED
        )
        # Optional: report composition
        edge_mask_after, center_mask_after = get_edge_center_masks(y_combined, grid_size=EDGE_GRID_SIZE)
        frac_edge = edge_mask_after.sum() / len(y_combined)
        print(f"  Achieved composition: edge={frac_edge:.3f}, center={1.0-frac_edge:.3f}")

    return X_combined, y_combined, orig_dimensions

# -------------------------------------------------------------------
# Training utilities
# -------------------------------------------------------------------
def compile_model(model, learning_rate):
    opt = tfa.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-5,
        clipnorm=0.5,
        epsilon=1e-8
    )
    model.compile(optimizer=opt, loss=combined_loss,
                  metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
    return model

def make_datasets(X, y, batch_size, augment=True):
    # Split
    val_split = 0.2
    n_val = int(len(X) * val_split)
    perm = np.random.permutation(len(X))
    train_idx, val_idx = perm[:-n_val], perm[-n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # tf.data pipelines
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_train = ds_train.shuffle(10000, seed=SEED, reshuffle_each_iteration=True)
    if augment:
        ds_train = ds_train.map(lambda x, y: (data_augmentation(x, training=True), y),
                                num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds_train, ds_val

def train_model(X, y, image_size, is_finetuning=False, existing_model=None, config=None, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)

    batch_size = config['batch_size']
    lr = config['learning_rate']
    epochs = config['epochs']
    min_lr = config['min_learning_rate']

    ds_train, ds_val = make_datasets(X, y, batch_size, augment=True)

    with strategy.scope():
        if is_finetuning:
            print("Preparing model for fine-tuning (unfreezing all layers)...")
            model = existing_model
            for layer in model.layers:
                layer.trainable = True
            model = compile_model(model, lr)
        else:
            print("Building new model for edge training...")
            model = build_enhanced_model(input_shape=(image_size, image_size, 3), trainable_tail=40)
            model = compile_model(model, lr)

    ckpt_path = os.path.join(model_dir, "edge_model.keras" if not is_finetuning else "final_model.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_mae', save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_mae', patience=15, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=5, min_lr=min_lr, verbose=1),
        keras.callbacks.CSVLogger(os.path.join(model_dir, 'training_log.csv'), append=is_finetuning)
    ]

    history = model.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=callbacks, verbose=1)
    return model, history

# -------------------------------------------------------------------
# Plot training curves
# -------------------------------------------------------------------
def plot_training_history(history, title_prefix=""):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title(f'{title_prefix} Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'], label='Train')
    plt.plot(history.history['val_mae'], label='Val')
    plt.title(f'{title_prefix} MAE')
    plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history['root_mean_squared_error'], label='Train')
    plt.plot(history.history['val_root_mean_squared_error'], label='Val')
    plt.title(f'{title_prefix} RMSE')
    plt.xlabel('Epoch'); plt.ylabel('RMSE'); plt.legend()

    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# Evaluation utilities
# -------------------------------------------------------------------
def prepare_evaluation_data(csv_path, video_path, image_size=256):
    """Load and preprocess evaluation data for a single source."""
    print(f"Preparing evaluation data from: {csv_path}")
    df = pd.read_csv(csv_path)
    frame_indices = df.iloc[:, 0].values
    coordinates = df.iloc[:, 1:3].values

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    X = np.zeros((len(frame_indices), image_size, image_size, 3), dtype=np.float32)
    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            if i > 0:
                X[i] = X[i-1]
            else:
                X[i] = 0.0
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        X[i] = frame
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(frame_indices)} frames")
    cap.release()

    # preprocess
    chunk = 1000
    for s in range(0, len(X), chunk):
        e = min(s + chunk, len(X))
        X[s:e] = preprocess_input(X[s:e])

    # normalize labels
    y = coordinates.astype(np.float32) / np.array([width, height], dtype=np.float32)
    y = np.clip(y, 1e-7, 1.0 - 1e-7)
    return X, y, (width, height)

def evaluate_model(model, X, y, orig_dimensions, batch_size=32, label=""):
    """Evaluate model performance with pixel error and 'axis accuracy'."""
    print(f"\nEvaluating {label} ...")
    y_pred = model.predict(X, batch_size=batch_size, verbose=1)

    y_pred_px = y_pred * np.array([orig_dimensions[0], orig_dimensions[1]])
    y_true_px = y * np.array([orig_dimensions[0], orig_dimensions[1]])

    pixel_errors = np.abs(y_pred_px - y_true_px)
    mean_err = np.mean(pixel_errors, axis=0)
    std_err = np.std(pixel_errors, axis=0)

    acc_x = (1.0 - mean_err[0] / orig_dimensions[0]) * 100.0
    acc_y = (1.0 - mean_err[1] / orig_dimensions[1]) * 100.0

    print(f"  Mean X Error (px): {mean_err[0]:.2f} ± {std_err[0]:.2f}")
    print(f"  Mean Y Error (px): {mean_err[1]:.2f} ± {std_err[1]:.2f}")
    print(f"  Accuracy X: {acc_x:.2f}%")
    print(f"  Accuracy Y: {acc_y:.2f}%")
    return acc_x, acc_y

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    csv_paths = [
        'drive/MyDrive/CNN/total_dataDCA.csv',
        'drive/MyDrive/processed/processed_YCA.csv',
        'drive/MyDrive/processed/processed_OCA.csv'
    ]
    video_paths = [
        'drive/MyDrive/CNN/scenevideoDCA.mp4',
        'drive/MyDrive/CNN/scenevideoYCA.mp4',
        'drive/MyDrive/CNN/scenevideoOCA.mp4'
    ]

    os.makedirs('models', exist_ok=True)

    # ---- Phase 1: Edge-focused training
    print("Step 1: Loading edge-focused multi-person dataset...")
    X_edge, y_edge, orig_dims = load_and_prepare_multi_person_data(
        csv_paths=csv_paths,
        video_paths=video_paths,
        num_samples=CONFIG['num_samples'],
        image_size=CONFIG['image_size'],
        edge_focused=True,
        edge_ratio=EDGE_TARGET_RATIO
    )

    print("Training on edge-rebalanced dataset...")
    edge_model, edge_history = train_model(
        X_edge, y_edge,
        image_size=CONFIG['image_size'],
        is_finetuning=False,
        existing_model=None,
        config=CONFIG,
        model_dir="models"
    )
    plot_training_history(edge_history, "Edge Training - ")
    edge_acc_x_train, edge_acc_y_train = evaluate_model(
        edge_model, X_edge, y_edge, orig_dims, batch_size=32, label="Edge-Training Data"
    )

    # ---- Phase 2: Fine-tuning on the original
    print("\nStep 2: Loading complete multi-person dataset for fine-tuning...")
    X_full, y_full, _ = load_and_prepare_multi_person_data(
        csv_paths=csv_paths,
        video_paths=video_paths,
        num_samples=FINETUNE_CONFIG['num_samples'],
        image_size=CONFIG['image_size'],
        edge_focused=False 
    )

    print("Fine-tuning on original distribution...")
    final_model, finetune_history = train_model(
        X_full, y_full,
        image_size=CONFIG['image_size'],
        is_finetuning=True,
        existing_model=edge_model,
        config=FINETUNE_CONFIG,
        model_dir="models"
    )
    plot_training_history(finetune_history, "Fine-tuning - ")

    # Save final model explicitly
    final_save_path = os.path.join("models", "final_multi_person_model.keras")
    final_model.save(final_save_path)
    print(f"Final model saved to: {final_save_path}")

    # ---- Per-subject evaluations ----
    # OCA
    X_eval, y_eval, dims = prepare_evaluation_data(
        'drive/MyDrive/processed/processed_OCA.csv',
        'drive/MyDrive/CNN/scenevideoOCA.mp4',
        image_size=CONFIG['image_size']
    )
    _ = evaluate_model(final_model, X_eval, y_eval, dims, batch_size=32, label="OCA evaluation")

    # YCA
    X_eval, y_eval, dims = prepare_evaluation_data(
        'drive/MyDrive/processed/processed_YCA.csv',
        'drive/MyDrive/CNN/scenevideoYCA.mp4',
        image_size=CONFIG['image_size']
    )
    _ = evaluate_model(final_model, X_eval, y_eval, dims, batch_size=32, label="YCA evaluation")

    # DCA
    X_eval, y_eval, dims = prepare_evaluation_data(
        'drive/MyDrive/CNN/total_dataDCA.csv',
        'drive/MyDrive/CNN/scenevideoDCA.mp4',
        image_size=CONFIG['image_size']
    )
    _ = evaluate_model(final_model, X_eval, y_eval, dims, batch_size=32, label="DCA evaluation")

    print("\nTraining Complete!")
