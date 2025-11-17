!pip install -q tensorflow_addons
!pip install -q opencv-python-headless

import os
import gc
import json
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L, preprocess_input
from tensorflow.keras.layers import Layer, Lambda, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Conv2D

# --------------------------------------------------------------------------------------
# TPU / Strategy Setup
# --------------------------------------------------------------------------------------
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("TPU detected and initialized")
    print(f"Number of accelerators: {strategy.num_replicas_in_sync}")
except Exception as e:
    print("No TPU detected. Falling back to CPU/GPU")
    print(f"(Info: {e})")
    strategy = tf.distribute.get_strategy()

# --------------------------------------------------------------------------------------
# Global Configuration
# --------------------------------------------------------------------------------------
CONFIG = {
    'num_samples': 27000,  
    'image_size': 224,
    'batch_size': 32 * 8,   
    'epochs': 100,
    'learning_rate': 2e-3,
    'min_learning_rate': 1e-6,
    'warm_up_epochs': 5
}

# --------------------------------------------------------------------------------------
# Data augmentation
# --------------------------------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.GaussianNoise(0.01)
], name="data_augmentation")

# --------------------------------------------------------------------------------------
# Coordinate Attention Layer
# --------------------------------------------------------------------------------------
class CoordinateAttention(Layer):
    def __init__(self, reduction_ratio=32, **kwargs):
        super(CoordinateAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        reduced_channels = max(channel // self.reduction_ratio, 8)

        self.down_conv = Conv2D(reduced_channels, 1, use_bias=False)
        self.bn1 = BatchNormalization()
        self.up_conv = Conv2D(channel, 1, use_bias=False)

    def call(self, inputs):
        x_h = Lambda(lambda x: tf.reduce_mean(x, axis=2, keepdims=True))(inputs)
        x_w = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(inputs)

        x_h = self.down_conv(x_h)
        x_w = self.down_conv(x_w)

        x_h = self.bn1(x_h)
        x_w = self.bn1(x_w)

        x_h = tf.nn.relu(x_h)
        x_w = tf.nn.relu(x_w)

        x_h = self.up_conv(x_h)
        x_w = self.up_conv(x_w)

        attention = tf.sigmoid(x_h + x_w)
        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config

# --------------------------------------------------------------------------------------
# Data Loading / Preparation
# --------------------------------------------------------------------------------------
def load_and_prepare_data(csv_path, video_path, num_samples=None):
    print(f"\nLoading and preparing data from:\n  CSV: {csv_path}\n  VIDEO: {video_path}")
    df = pd.read_csv(csv_path)
    frame_indices = df.iloc[:, 0].values
    coordinates = df.iloc[:, 1:3].values

    if num_samples:
        num_samples = min(num_samples, len(frame_indices))
        indices = np.random.choice(len(frame_indices), num_samples, replace=False)
        frame_indices = frame_indices[indices]
        coordinates = coordinates[indices]

    video = cv2.VideoCapture(video_path)
    orig_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = len(frame_indices)

    # Pre-allocate arrays
    X = np.zeros((total_frames, CONFIG['image_size'], CONFIG['image_size'], 3), dtype=np.float32)
    y = np.zeros((total_frames, 2), dtype=np.float32)

    for i, (idx, coord) in enumerate(zip(frame_indices, coordinates)):
        video.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = video.read()
        if not ret:
            # If frame can't be read, keep zero; but inform:
            if (i + 1) % 100 == 0 or i == total_frames - 1:
                print(f"Warning: could not read frame {idx}, filled with zeros.")
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (CONFIG['image_size'], CONFIG['image_size']))
        X[i] = frame
        y[i] = coord

        if (i + 1) % 1000 == 0 or (i + 1) == total_frames:
            print(f"Processed {i + 1}/{total_frames} frames")

    video.release()

    # Preprocess in manageable chunks
    batch_size = 1000
    for i in range(0, len(X), batch_size):
        end_idx = min(i + batch_size, len(X))
        X[i:end_idx] = preprocess_input(X[i:end_idx])

    # Normalize coordinates to [0,1] by original video dimensions
    y = y / np.array([orig_width, orig_height], dtype=np.float32)

    return X, y, (orig_width, orig_height)

# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def build_model():
    input_layer = keras.Input(shape=(CONFIG['image_size'], CONFIG['image_size'], 3))

    backbone = EfficientNetV2L(
        include_top=False,
        weights='imagenet',
        input_tensor=input_layer
    )

    # Freeze early layers
    for layer in backbone.layers[:-30]:
        layer.trainable = False

    x = backbone.output
    x = CoordinateAttention()(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(512, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    outputs = Dense(2, activation='sigmoid')(x)

    return keras.Model(input_layer, outputs)

# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------
def plot_training_history(history, tag):
    plt.figure(figsize=(15, 5))

    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss ({tag})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'MAE ({tag})')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    # Plot RMSE
    plt.subplot(1, 3, 3)
    plt.plot(history.history['root_mean_squared_error'], label='Training RMSE')
    plt.plot(history.history['val_root_mean_squared_error'], label='Validation RMSE')
    plt.title(f'RMSE ({tag})')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()

    plt.tight_layout()
    out_path = f'training_history_{tag}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved training history plot to: {out_path}")

# --------------------------------------------------------------------------------------
# Training (single dataset)
# --------------------------------------------------------------------------------------
def train_model(csv_path, video_path, tag):
    # Ensure output directories exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs(f'logs/{tag}', exist_ok=True)

    # Load and prepare data
    X, y, (orig_width, orig_height) = load_and_prepare_data(
        csv_path, video_path, CONFIG['num_samples'])

    # Split data
    val_split = 0.2
    num_val = int(len(X) * val_split)
    indices = np.random.permutation(len(X))

    train_indices = indices[:-num_val]
    val_indices = indices[-num_val:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    # Create datasets with augmentation
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(10000)
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.batch(CONFIG['batch_size'])
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(CONFIG['batch_size'])

    # Build and compile model
    with strategy.scope():
        model = build_model()
        model.compile(
            optimizer=tfa.optimizers.AdamW(
                learning_rate=CONFIG['learning_rate'],
                weight_decay=1e-4
            ),
            loss=tf.keras.losses.Huber(delta=1.0),
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
        )

    # Callbacks (tagged)
    best_model_path = f'best_model_{tag}.keras'
    final_model_path = f'final_model_{tag}.keras'
    csv_log_path = f'training_log_{tag}.csv'
    tb_log_dir = f'./logs/{tag}'

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=7,
            min_lr=CONFIG['min_learning_rate'],
            verbose=1
        ),
        keras.callbacks.CSVLogger(csv_log_path),
        keras.callbacks.TensorBoard(log_dir=tb_log_dir)
    ]

    # Train the model
    print(f"\n=== Training start: {tag} ===")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    print(f"=== Training end: {tag} ===\n")

    # Save final model (in addition to best checkpoint)
    model.save(final_model_path)
    print(f"Saved final model to: {final_model_path}")

    # Evaluate
    print("\nEvaluating model...")
    y_pred = model.predict(X_val, verbose=0)
    y_pred_pixels = y_pred * np.array([orig_width, orig_height], dtype=np.float32)
    y_val_pixels = y_val * np.array([orig_width, orig_height], dtype=np.float32)

    pixel_errors = np.abs(y_pred_pixels - y_val_pixels)
    mean_error = np.mean(pixel_errors, axis=0)
    std_error = np.std(pixel_errors, axis=0)

    acc_x = (1.0 - mean_error[0] / orig_width)
    acc_y = (1.0 - mean_error[1] / orig_height)

    print("\nFinal Results:")
    print(f"[{tag}] Mean X Error (pixels): {mean_error[0]:.2f} ± {std_error[0]:.2f}")
    print(f"[{tag}] Mean Y Error (pixels): {mean_error[1]:.2f} ± {std_error[1]:.2f}")
    print(f"[{tag}] Accuracy X: {acc_x:.4%}")
    print(f"[{tag}] Accuracy Y: {acc_y:.4%}")

    # Save evaluation summary
    eval_summary = {
        "tag": tag,
        "orig_width": int(orig_width),
        "orig_height": int(orig_height),
        "mean_x_error_px": float(mean_error[0]),
        "std_x_error_px": float(std_error[0]),
        "mean_y_error_px": float(mean_error[1]),
        "std_y_error_px": float(std_error[1]),
        "accuracy_x": float(acc_x),
        "accuracy_y": float(acc_y),
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "csv_log_path": csv_log_path,
        "tensorboard_log_dir": tb_log_dir
    }
    with open(f"evaluation_{tag}.json", "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"Saved evaluation summary to: evaluation_{tag}.json")

    # Plot training history
    plot_training_history(history, tag)

    # Cleanup large arrays to free memory before next run
    del X, y, X_train, y_train, X_val, y_val, train_ds, val_ds
    gc.collect()

    return model, history

# --------------------------------------------------------------------------------------
# Main: Train 3 models back-to-back on specified datasets
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Dataset list with tags and paths
    datasets = [
        {
            "tag": "YCA",
            "csv_path": "drive/MyDrive/CNN/total_dataYCA.csv",
            "video_path": "drive/MyDrive/CNN/scenevideoYCA.mp4"
        },
        {
            "tag": "DCA",
            "csv_path": "drive/MyDrive/CNN/total_dataDCA.csv",
            "video_path": "drive/MyDrive/CNN/scenevideoDCA.mp4"
        },
        {
            "tag": "OCA",
            "csv_path": "drive/MyDrive/CNN/total_dataOCA.csv",
            "video_path": "drive/MyDrive/CNN/scenevideoOCA.mp4"
        }
    ]

    np.random.seed(42)
    tf.random.set_seed(42)

    trained_artifacts = {}
    for ds in datasets:
        tag = ds["tag"]
        csv_path = ds["csv_path"]
        video_path = ds["video_path"]

        print(f"\n==============================")
        print(f" Starting training for: {tag}")
        print(f"==============================\n")

        model, history = train_model(csv_path, video_path, tag)
        trained_artifacts[tag] = {
            "model_path_best": f"best_model_{tag}.keras",
            "model_path_final": f"final_model_{tag}.keras",
            "history_keys": list(history.history.keys())
        }

        # Clear Keras session between runs to reduce memory pressure
        keras.backend.clear_session()
        gc.collect()

    print("\nAll trainings completed.")
    print("Saved models:")
    print(" - best_model_YCA.keras / final_model_YCA.keras")
    print(" - best_model_DCA.keras / final_model_DCA.keras")
    print(" - best_model_OCA.keras / final_model_OCA.keras")
