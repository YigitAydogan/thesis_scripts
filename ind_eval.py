import os
import json
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, Lambda, BatchNormalization, Conv2D
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
IMAGE_SIZE = 224          
EVAL_BATCH_SIZE = 128     
MODEL_VARIANT = "best"    

# --------------------------------------------------------------------------------------
# Custom layer used in the trained models
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
# Data preparation for evaluation
# --------------------------------------------------------------------------------------
def prepare_evaluation_data(csv_path, video_path, image_size=IMAGE_SIZE):
    """Load and preprocess the evaluation dataset"""
    print(f"\nLoading evaluation data:\n  CSV: {csv_path}\n  VIDEO: {video_path}")
    df = pd.read_csv(csv_path)
    frame_indices = df.iloc[:, 0].values
    coordinates = df.iloc[:, 1:3].values

    # Open video and get dimensions
    video = cv2.VideoCapture(video_path)
    orig_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize array for frames
    X = np.zeros((len(frame_indices), image_size, image_size, 3), dtype=np.float32)
    y = coordinates.astype(np.float32)

    # Load and preprocess frames
    for i, idx in enumerate(frame_indices):
        video.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size))
            X[i] = frame
        else:
            # If a frame cannot be read, leave zeros; log occasionally
            if (i + 1) % 200 == 0 or (i + 1) == len(frame_indices):
                print(f"Warning: could not read frame {idx}; left as zeros.")
        if (i + 1) % 1000 == 0 or (i + 1) == len(frame_indices):
            print(f"Processed {i + 1}/{len(frame_indices)} frames")

    video.release()

    # Preprocess images in batches
    batch_size = 1000
    for i in range(0, len(X), batch_size):
        end_idx = min(i + batch_size, len(X))
        X[i:end_idx] = preprocess_input(X[i:end_idx])

    # Normalize coordinates
    y = y / np.array([orig_width, orig_height], dtype=np.float32)
    y = np.clip(y, 1e-7, 1.0 - 1e-7)

    return X, y, (orig_width, orig_height)

# --------------------------------------------------------------------------------------
# Model loading and evaluation
# --------------------------------------------------------------------------------------
def load_model_safely(path):
    """Load a saved Keras model with custom objects, without compiling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    print(f"Loading model: {path}")
    model = keras.models.load_model(
        path,
        custom_objects={"CoordinateAttention": CoordinateAttention},
        compile=False
    )
    return model

def evaluate_model(model, X, y, orig_dimensions, batch_size=EVAL_BATCH_SIZE):
    """Evaluate model performance and return metrics dictionary."""
    print("Running predictions...")
    y_pred = model.predict(X, batch_size=batch_size, verbose=1)

    # Convert predictions back to pixel coordinates
    w, h = float(orig_dimensions[0]), float(orig_dimensions[1])
    y_pred_pixels = y_pred * np.array([w, h], dtype=np.float32)
    y_true_pixels = y * np.array([w, h], dtype=np.float32)

    # Calculate errors
    pixel_errors = np.abs(y_pred_pixels - y_true_pixels)
    mean_error = np.mean(pixel_errors, axis=0)
    std_error = np.std(pixel_errors, axis=0)

    # Calculate accuracy
    accuracy_x = (1.0 - mean_error[0] / w) * 100.0
    accuracy_y = (1.0 - mean_error[1] / h) * 100.0
    avg_accuracy = (accuracy_x + accuracy_y) / 2.0

    # Print results
    print("Evaluation Results:")
    print(f"Mean X Error (pixels): {mean_error[0]:.2f} ± {std_error[0]:.2f}")
    print(f"Mean Y Error (pixels): {mean_error[1]:.2f} ± {std_error[1]:.2f}")
    print(f"Accuracy X: {accuracy_x:.2f}%")
    print(f"Accuracy Y: {accuracy_y:.2f}%")
    print(f"Average Accuracy: {avg_accuracy:.2f}%\n")

    return {
        "mean_x_error_px": float(mean_error[0]),
        "std_x_error_px": float(std_error[0]),
        "mean_y_error_px": float(mean_error[1]),
        "std_y_error_px": float(std_error[1]),
        "accuracy_x_pct": float(accuracy_x),
        "accuracy_y_pct": float(accuracy_y),
        "avg_accuracy_pct": float(avg_accuracy),
    }

# --------------------------------------------------------------------------------------
# Define datasets and model files
# --------------------------------------------------------------------------------------
DATASETS = [
    {
        "tag": "YCA",
        "csv_path": "drive/MyDrive/processed/processed_YCA.csv",
        "video_path": "drive/MyDrive/CNN/scenevideoYCA.mp4",
    },
    {
        "tag": "DCA",
        "csv_path": "drive/MyDrive/processed/processed_DCA.csv",
        "video_path": "drive/MyDrive/CNN/scenevideoDCA.mp4",
    },
    {
        "tag": "OCA",
        "csv_path": "drive/MyDrive/processed/processed_OCA.csv",
        "video_path": "drive/MyDrive/CNN/scenevideoOCA.mp4",
    },
]

def resolve_model_path(tag, variant=MODEL_VARIANT):
    """Return an existing model path for the given tag and variant, with a safe fallback."""
    primary = f"{'best_model' if variant == 'best' else 'final_model'}_{tag}.keras"
    fallback = f"{'final_model' if variant == 'best' else 'best_model'}_{tag}.keras"
    if os.path.exists(primary):
        return primary
    if os.path.exists(fallback):
        print(f"Note: '{primary}' not found. Falling back to '{fallback}'.")
        return fallback
    raise FileNotFoundError(f"Neither '{primary}' nor '{fallback}' exists in the current directory.")

# --------------------------------------------------------------------------------------
# Main: Perform the 3x3 cross-evaluation and save results
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Prepare and cache all evaluation datasets once
    cached_data = {}
    for ds in DATASETS:
        tag = ds["tag"]
        X_eval, y_eval, orig_dims = prepare_evaluation_data(ds["csv_path"], ds["video_path"], IMAGE_SIZE)
        cached_data[tag] = (X_eval, y_eval, orig_dims)

    # Load each model and evaluate on all datasets (9 evaluations total)
    results = []
    for model_tag in ["YCA", "DCA", "OCA"]:
        model_path = resolve_model_path(model_tag, MODEL_VARIANT)
        model = load_model_safely(model_path)

        for data_tag in ["YCA", "DCA", "OCA"]:
            print(f"\n===== Evaluating model [{model_tag}] on dataset [{data_tag}] =====")
            X_eval, y_eval, orig_dims = cached_data[data_tag]
            metrics = evaluate_model(model, X_eval, y_eval, orig_dims, batch_size=EVAL_BATCH_SIZE)
            row = {
                "model_tag": model_tag,
                "dataset_tag": data_tag,
                "model_path": model_path,
                **metrics,
            }
            results.append(row)

        # Clear session between models to free memory
        keras.backend.clear_session()

    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df[
        [
            "model_tag",
            "dataset_tag",
            "model_path",
            "mean_x_error_px",
            "std_x_error_px",
            "mean_y_error_px",
            "std_y_error_px",
            "accuracy_x_pct",
            "accuracy_y_pct",
            "avg_accuracy_pct",
        ]
    ]
    results_df.to_csv("cross_eval_results.csv", index=False)
    with open("cross_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n================ Cross-Evaluation Summary ================\n")
    print(results_df.to_string(index=False))
    print("\nSaved results to 'cross_eval_results.csv' and 'cross_eval_results.json'.")
