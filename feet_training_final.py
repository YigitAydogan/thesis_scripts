# Install necessary packages
!pip install -q tensorflow-addons

# Import necessary libraries
import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L, preprocess_input
from tensorflow.keras.layers import Layer, Lambda, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Conv2D

# ========================== TPU Setup ==========================
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("TPU detected and initialized")
    print(f"Number of accelerators: {strategy.num_replicas_in_sync}")
except ValueError:
    # If TPU is not available, fallback to default strategy
    strategy = tf.distribute.get_strategy()
    print("TPU not detected. Using default strategy.")
    print(f"Number of accelerators: {strategy.num_replicas_in_sync}")

SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ================== Training configuration =====================
# Test configuration with small sample sizes
TEST_CONFIG = {
    'num_samples': 300,  # 100 samples per person
    'image_size': 256,
    'batch_size': 16,    # Reduced batch size for testing
    'epochs': 5,         # Reduced epochs for testing
    'learning_rate': 1e-4,
    'min_learning_rate': 1e-6,
    'warmup_epochs': 2
}

TEST_FINETUNE_CONFIG = {
    'num_samples': 300,  # 100 samples per person
    'batch_size': 16,
    'epochs': 3,         # Reduced epochs for testing
    'learning_rate': 1e-4,
    'min_learning_rate': 1e-7
}

# Full training configuration
FULL_CONFIG = {
    'num_samples': 1538*3,  
    'image_size': 256,
    'batch_size': 16 * 8,
    'epochs': 170,
    'learning_rate': 1e-4,
    'min_learning_rate': 1e-6,
    'warmup_epochs': 5
}

FULL_FINETUNE_CONFIG = {
    'num_samples': 1538*3,  
    'batch_size': 16 * 8,
    'epochs': 90,
    'learning_rate': 1e-4,
    'min_learning_rate': 1e-7
}

# Use FULL_CONFIG for actual training, switch to TEST_CONFIG for testing
CONFIG = FULL_CONFIG
FINETUNE_CONFIG = FULL_FINETUNE_CONFIG

# ===== Edge/Center definition (3×3) & target ratio for edge-focused phase =====
EDGE_GRID_SIZE = 3           # 3x3 grid
EDGE_TARGET_RATIO = 0.70     # target mix during edge-focused loading (70% edge / 30% center)

# ======================= Data augmentation =====================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.GaussianNoise(0.01),
    tf.keras.layers.RandomZoom(
        height_factor=(-0.1, 0.1),
        width_factor=(-0.1, 0.1),
        fill_mode='constant'
    )
])

# ================== Custom Layers and Model ====================
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

        self.spatial_conv = Conv2D(self.groups, 7, padding='same', use_bias=False,
                                   kernel_initializer='he_normal')
        self.spatial_bn = BatchNormalization()

        channel_per_group = channel // self.groups
        self.final_convs = [Conv2D(channel_per_group, 1, use_bias=False,
                                   kernel_initializer='he_normal') for _ in range(self.groups)]

    def call(self, inputs):
        channel = inputs.shape[-1]
        _ = channel // self.groups

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
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "groups": self.groups
        })
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
        config.update({
            "units": self.units,
            "dropout_rate": self.dropout_rate
        })
        return config

def build_enhanced_model(input_shape=(256, 256, 3), backbone='efficientnetv2l'):
    input_layer = keras.Input(shape=input_shape)

    if backbone == 'efficientnetv2l':
        backbone_model = keras.applications.efficientnet_v2.EfficientNetV2L(
            include_top=False,
            weights='imagenet',
            input_tensor=input_layer
        )

    # Freeze all layers except the last 40
    for layer in backbone_model.layers[:-40]:
        layer.trainable = False

    x = backbone_model.output
    x = GroupedCoordinateAttention(reduction_ratio=16, groups=4)(x)
    x = GlobalAveragePooling2D()(x)

    x = PreActResidualBlock(512, dropout_rate=0.4)(x)
    x = PreActResidualBlock(256, dropout_rate=0.3)(x)
    x = PreActResidualBlock(128, dropout_rate=0.2)(x)

    outputs = Dense(2, activation='sigmoid', kernel_initializer='glorot_normal')(x)

    return keras.Model(input_layer, outputs)

# ===================== Loss (single definition) ==================
def combined_loss(y_true, y_pred):
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    huber = tf.keras.losses.Huber(
        delta=1.0,
        reduction=tf.keras.losses.Reduction.NONE
    )(y_true, y_pred)

    mse = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.NONE
    )(y_true, y_pred)

    combined = 0.7 * tf.reduce_mean(huber) + 0.3 * tf.reduce_mean(mse)
    return tf.where(tf.math.is_nan(combined), 1e3, combined)

def compute_grid_indices(norm_xy, grid_size=3):
    """
    norm_xy: (N,2) with normalized x,y in [0,1]
    Returns x_idx, y_idx in {0,...,grid_size-1}
    """
    x = np.clip(norm_xy[:, 0], 0.0, 1.0 - 1e-8)
    y = np.clip(norm_xy[:, 1], 0.0, 1.0 - 1e-8)
    edges = np.linspace(0, 1, grid_size + 1)
    x_idx = np.digitize(x, edges) - 1
    y_idx = np.digitize(y, edges) - 1
    return x_idx, y_idx

def get_edge_center_masks(norm_xy, grid_size=3):
    """
    Center = middle cell only (for 3x3: (1,1)); Edge = other 8 cells.
    Returns boolean masks (edge_mask, center_mask).
    """
    assert grid_size == 3, "This implementation assumes a 3x3 grid."
    x_idx, y_idx = compute_grid_indices(norm_xy, grid_size)
    mid = grid_size // 2  # =1 for 3x3
    center_mask = (x_idx == mid) & (y_idx == mid)
    edge_mask = ~center_mask
    return edge_mask, center_mask

def rebalance_edge_center(X, y, edge_ratio=0.7, grid_size=3, seed=SEED):
    """
    Rebalances (with replacement) to reach the desired edge_ratio while preserving dataset size.
    X: images (N, H, W, 3)
    y: normalized coords (N, 2 in [0,1])
    """
    rng = np.random.default_rng(seed)
    edge_mask, center_mask = get_edge_center_masks(y, grid_size=grid_size)

    edge_idx = np.where(edge_mask)[0]
    center_idx = np.where(center_mask)[0]

    total = len(y)
    n_edge_target = int(total * edge_ratio)
    n_center_target = total - n_edge_target

    chosen_edge = rng.choice(edge_idx, size=n_edge_target, replace=True) if len(edge_idx) > 0 else np.array([], dtype=int)
    chosen_center = rng.choice(center_idx, size=n_center_target, replace=True) if len(center_idx) > 0 else np.array([], dtype=int)

    chosen = np.concatenate([chosen_edge, chosen_center])
    rng.shuffle(chosen)
    return X[chosen], y[chosen]

# ==================== Evaluation & plotting =====================
def evaluate_model(model, X, y, orig_dimensions, phase=""):
    print(f"\nEvaluating model after {phase}...")
    y_pred = model.predict(X, batch_size=32)

    y_pred_pixels = y_pred * np.array([orig_dimensions[0], orig_dimensions[1]])
    y_true_pixels = y * np.array([orig_dimensions[0], orig_dimensions[1]])

    pixel_errors = np.abs(y_pred_pixels - y_true_pixels)
    mean_error = np.mean(pixel_errors, axis=0)
    std_error = np.std(pixel_errors, axis=0)

    accuracy_x = (1 - mean_error[0]/orig_dimensions[0]) * 100
    accuracy_y = (1 - mean_error[1]/orig_dimensions[1]) * 100

    print(f"\nResults after {phase}:")
    print(f"Mean X Error (pixels): {mean_error[0]:.2f} ± {std_error[0]:.2f}")
    print(f"Mean Y Error (pixels): {mean_error[1]:.2f} ± {std_error[1]:.2f}")
    print(f"Accuracy X: {accuracy_x:.2f}%")
    print(f"Accuracy Y: {accuracy_y:.2f}%")

    return accuracy_x, accuracy_y

def plot_training_history(history, title_prefix=""):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title_prefix} Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{title_prefix} Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history['root_mean_squared_error'], label='Training RMSE')
    plt.plot(history.history['val_root_mean_squared_error'], label='Validation RMSE')
    plt.title(f'{title_prefix} Model RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()

    plt.tight_layout()
    plt.show()

# =================== Data loading (with segments) =================
def load_and_prepare_multi_person_data(csv_paths, video_paths, frame_segments, num_samples=None, edge_focused=False):
    """
    Load and prepare multi-person data with specified frame segments for each video.

    Args:
        csv_paths (list of str): List of CSV file paths.
        video_paths (list of str): List of video file paths.
        frame_segments (list of list of tuples):
            Each sublist corresponds to a video and contains tuples of (start_frame, end_frame).
        num_samples (int, optional): Number of samples to select per person. Defaults to None.
        edge_focused (bool, optional): If True, rebalance to emphasize edge samples via 3x3 logic.

    Returns:
        tuple: (X_combined, y_combined, orig_dimensions)
    """
    print("Loading and preparing multi-person data...")

    all_X = []
    all_y = []
    orig_dimensions = None

    for i, (csv_path, video_path) in enumerate(zip(csv_paths, video_paths)):
        segments = frame_segments[i]  # List of (start, end) tuples for current video
        print(f"Processing data from: {csv_path} with segments: {segments}")
        df = pd.read_csv(csv_path)
        frame_indices = df.iloc[:, 0].values
        coordinates = df.iloc[:, 1:3].values

        # Filter frame_indices and coordinates based on frame_segments
        mask = np.zeros_like(frame_indices, dtype=bool)
        for start, end in segments:
            mask |= (frame_indices >= start) & (frame_indices <= end)
        filtered_frame_indices = frame_indices[mask]
        filtered_coordinates = coordinates[mask]

        # Subsample per-person if requested
        if num_samples:
            samples_per_person = num_samples // len(csv_paths)
            if samples_per_person > len(filtered_frame_indices):
                print(f"Warning: Requested {samples_per_person} samples, but only {len(filtered_frame_indices)} available.")
                samples_per_person = len(filtered_frame_indices)
            selected_indices = np.random.choice(
                len(filtered_frame_indices),
                samples_per_person,
                replace=False
            )
            filtered_frame_indices = filtered_frame_indices[selected_indices]
            filtered_coordinates = filtered_coordinates[selected_indices]

        video = cv2.VideoCapture(video_path)
        if orig_dimensions is None:
            orig_dimensions = (
                int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )

        X = np.zeros((len(filtered_frame_indices), CONFIG['image_size'], CONFIG['image_size'], 3),
                     dtype=np.float32)

        for j, (idx, coord) in enumerate(zip(filtered_frame_indices, filtered_coordinates)):
            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (CONFIG['image_size'], CONFIG['image_size']))
                X[j] = frame
                if (j + 1) % 1000 == 0:
                    print(f"Processed {j + 1}/{len(filtered_frame_indices)} frames for current video")
            else:
                print(f"Warning: Frame {idx} could not be read from {video_path}.")
                if j > 0:
                    X[j] = X[j-1]
                else:
                    X[j] = 0.0

        video.release()

        # Preprocess in batches
        batch_size = 1000
        for j in range(0, len(X), batch_size):
            end_idx = min(j + batch_size, len(X))
            X[j:end_idx] = preprocess_input(X[j:end_idx])

        # Normalize coordinates to [0,1] using the (first) video's dimensions
        y = filtered_coordinates / np.array([orig_dimensions[0], orig_dimensions[1]])
        y = np.clip(y, 1e-7, 1.0 - 1e-7)

        if edge_focused and len(y) > 0:
            X, y = rebalance_edge_center(X, y, edge_ratio=EDGE_TARGET_RATIO, grid_size=EDGE_GRID_SIZE, seed=SEED)

        all_X.append(X)
        all_y.append(y)

    # Combine all datasets
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    # Shuffle combined dataset
    shuffle_idx = np.random.permutation(len(X_combined))
    X_combined = X_combined[shuffle_idx]
    y_combined = y_combined[shuffle_idx]

    print(f"Total samples after combining: {X_combined.shape[0]}")
    return X_combined, y_combined, orig_dimensions

# ============================ Training ==========================
def train_model(X, y, orig_dimensions, is_finetuning=False, existing_model=None):
    """
    Train or fine-tune the model.

    Args:
        X (np.array): Input data.
        y (np.array): Target coordinates.
        orig_dimensions (tuple): Original video dimensions.
        is_finetuning (bool, optional): Whether to fine-tune an existing model. Defaults to False.
        existing_model (keras.Model, optional): Existing model to fine-tune. Defaults to None.

    Returns:
        tuple: (trained_model, training_history)
    """
    val_split = 0.2
    num_val = int(len(X) * val_split)
    indices = np.random.permutation(len(X))

    train_indices = indices[:-num_val]
    val_indices = indices[-num_val:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(10000, seed=SEED, reshuffle_each_iteration=True)
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.batch(CONFIG['batch_size'] if not is_finetuning else FINETUNE_CONFIG['batch_size'])
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(
        CONFIG['batch_size'] if not is_finetuning else FINETUNE_CONFIG['batch_size']
    ).prefetch(tf.data.AUTOTUNE)

    with strategy.scope():
        if is_finetuning:
            print("Preparing model for fine-tuning...")
            model = existing_model
            for layer in model.layers:
                layer.trainable = True
            learning_rate = FINETUNE_CONFIG['learning_rate']
            epochs = FINETUNE_CONFIG['epochs']
        else:
            print("Building new model for edge training...")
            model = build_enhanced_model(input_shape=(CONFIG['image_size'], CONFIG['image_size'], 3))
            learning_rate = CONFIG['learning_rate']
            epochs = CONFIG['epochs']

        model.compile(
            optimizer=tfa.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=1e-5,
                clipnorm=0.5,
                epsilon=1e-8
            ),
            loss=combined_loss,
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
        )

    model_name = 'models/edge_model.keras' if not is_finetuning else 'models/final_model.keras'
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_name,
            monitor='val_mae',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=5,
            min_lr=CONFIG['min_learning_rate'] if not is_finetuning else FINETUNE_CONFIG['min_learning_rate'],
            verbose=1
        ),
        keras.callbacks.CSVLogger('training_log.csv', append=is_finetuning)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    return model, history

# ======================= Evaluation helpers =====================
def prepare_evaluation_data(csv_path, video_path, frame_segments, image_size=256):
    """
    Load and preprocess the evaluation dataset with specified frame segments.

    Args:
        csv_path (str): Path to the CSV file.
        video_path (str): Path to the video file.
        frame_segments (list of tuples): List of (start_frame, end_frame) tuples.
        image_size (int, optional): Desired image size. Defaults to 256.

    Returns:
        tuple: (X_eval, y_eval, (orig_width, orig_height))
    """
    print(f"\nLoading evaluation data from {csv_path} with segments: {frame_segments}")
    df = pd.read_csv(csv_path)
    frame_indices = df.iloc[:, 0].values
    coordinates = df.iloc[:, 1:3].values

    # Filter frame_indices and coordinates based on frame_segments
    mask = np.zeros_like(frame_indices, dtype=bool)
    for start, end in frame_segments:
        mask |= (frame_indices >= start) & (frame_indices <= end)
    filtered_frame_indices = frame_indices[mask]
    filtered_coordinates = coordinates[mask]

    video = cv2.VideoCapture(video_path)
    orig_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    X = np.zeros((len(filtered_frame_indices), image_size, image_size, 3), dtype=np.float32)
    y = filtered_coordinates.copy()

    for i, idx in enumerate(filtered_frame_indices):
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size))
            X[i] = frame
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(filtered_frame_indices)} frames for evaluation")
        else:
            print(f"Warning: Frame {idx} could not be read from {video_path}.")
            if i > 0:
                X[i] = X[i-1]
            else:
                X[i] = 0.0

    video.release()

    # Preprocess images in batches
    batch_size = 1000
    for i in range(0, len(X), batch_size):
        end_idx = min(i + batch_size, len(X))
        X[i:end_idx] = preprocess_input(X[i:end_idx])

    # Normalize coordinates
    y = y / np.array([orig_width, orig_height])
    y = np.clip(y, 1e-7, 1.0 - 1e-7)

    print(f"Total evaluation samples: {X.shape[0]}")
    return X, y, (orig_width, orig_height)

def evaluate_on_additional_dataset(model, csv_path, video_path, frame_segments):
    """
    Evaluate the model on an additional dataset with specified frame segments.
    """
    X_eval, y_eval, orig_dimensions = prepare_evaluation_data(csv_path, video_path, frame_segments)
    evaluate_model(model, X_eval, y_eval, orig_dimensions, phase="additional evaluation")

# ============================ Main ==============================
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

    # Define frame segments for each video
    # Each sublist contains tuples of (start_frame, end_frame) for that video
    frame_segments = [
        [(898, 1871), (26347, 27195)],        # Segments for scenevideoDCA.mp4
        [(25972, 26821), (34795, 35528)],     # Segments for scenevideoYCA.mp4
        [(1272, 2120), (22604, 23453)]        # Segments for scenevideoOCA.mp4
    ]

    # Ensure that the number of frame_segments matches the number of videos
    assert len(frame_segments) == len(video_paths), "frame_segments must match the number of videos."

    # Create output directory for saved models and logs
    os.makedirs('models', exist_ok=True)

    # -------- Phase 1: Edge-focused training --------
    print("Step 1: Loading edge-focused multi-person dataset...")
    X_edge, y_edge, orig_dimensions = load_and_prepare_multi_person_data(
        csv_paths,
        video_paths,
        frame_segments, 
        CONFIG['num_samples'],
        edge_focused=True
    )

    print("Training on multi-person edge cases...")
    edge_model, edge_history = train_model(X_edge, y_edge, orig_dimensions)
    plot_training_history(edge_history, "Edge Training - ")
    edge_acc_x, edge_acc_y = evaluate_model(edge_model, X_edge, y_edge, orig_dimensions, "edge training")

    # -------------------- Phase 2: Fine-tuning -------------------
    print("\nStep 2: Loading complete multi-person dataset for fine-tuning...")
    X_complete, y_complete, _ = load_and_prepare_multi_person_data(
        csv_paths,
        video_paths,
        frame_segments,  # Pass frame_segments here
        FINETUNE_CONFIG['num_samples'],
        edge_focused=False
    )

    print("Fine-tuning on complete multi-person dataset...")
    final_model, finetune_history = train_model(
        X_complete,
        y_complete,
        orig_dimensions,
        is_finetuning=True,
        existing_model=edge_model
    )
    plot_training_history(finetune_history, "Fine-tuning - ")

    # Evaluate final model on complete dataset
    final_acc_x, final_acc_y = evaluate_model(
        final_model,
        X_complete,
        y_complete,
        orig_dimensions,
        "fine-tuning"
    )

    # Save final model
    final_model.save('models/final_multi_person_model.keras')

    print("\nTraining Complete!")
    print("\nOverall Results:")
    print(f"Edge-case accuracy: X={edge_acc_x:.2f}%, Y={edge_acc_y:.2f}%")
    print(f"Final accuracy: X={final_acc_x:.2f}%, Y={final_acc_y:.2f}%")

    # ==================== Evaluation on Additional Datasets ====================
    additional_evaluation_datasets = [
        {
            'csv_path': 'drive/MyDrive/processed/processed_OCA.csv',
            'video_path': 'drive/MyDrive/CNN/scenevideoOCA.mp4',
            'frame_segments': [(100, 600), (1100, 1600)]
        },
        {
            'csv_path': 'drive/MyDrive/processed/processed_YCA.csv',
            'video_path': 'drive/MyDrive/CNN/scenevideoYCA.mp4',
            'frame_segments': [(200, 700), (1200, 1700)]
        },
        {
            'csv_path': 'drive/MyDrive/CNN/total_dataDCA.csv',
            'video_path': 'drive/MyDrive/CNN/scenevideoDCA.mp4',
            'frame_segments': [(100, 500), (1000, 1500)]
        }
    ]

    for dataset in additional_evaluation_datasets:
        evaluate_on_additional_dataset(
            final_model,
            dataset['csv_path'],
            dataset['video_path'],
            dataset['frame_segments']
        )
