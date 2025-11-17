def prepare_evaluation_data(csv_path, video_path, image_size=256, num_frames=None):
    """Load and preprocess the evaluation dataset"""
    print("Loading evaluation data...")
    df = pd.read_csv(csv_path)

    # Limit frames if specified
    if num_frames is not None:
        df = df.head(num_frames)

    frame_indices = df.iloc[:, 0].values
    coordinates = df.iloc[:, 1:3].values

    # Open video and get dimensions
    video = cv2.VideoCapture(video_path)
    orig_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize array for frames
    X = np.zeros((len(frame_indices), image_size, image_size, 3), dtype=np.float32)
    y = coordinates.copy()

    # Load and preprocess frames
    for i, idx in enumerate(frame_indices):
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size))
            X[i] = frame
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(frame_indices)} frames")

    video.release()

    # Preprocess images in batches
    batch_size = 1000
    for i in range(0, len(X), batch_size):
        end_idx = min(i + batch_size, len(X))
        X[i:end_idx] = preprocess_input(X[i:end_idx])

    # Normalize coordinates
    y = y / np.array([orig_width, orig_height])
    y = np.clip(y, 1e-7, 1.0 - 1e-7)

    return X, y, (orig_width, orig_height)

def evaluate_model(model, X, y, orig_dimensions):
    """Evaluate model performance with various visualizations"""
    print("\nEvaluating model...")
    y_pred = model.predict(X, batch_size=32)

    # Convert predictions back to pixel coordinates
    y_pred_pixels = y_pred * np.array([orig_dimensions[0], orig_dimensions[1]])
    y_true_pixels = y * np.array([orig_dimensions[0], orig_dimensions[1]])

    # Calculate errors
    pixel_errors = np.abs(y_pred_pixels - y_true_pixels)
    mean_error = np.mean(pixel_errors, axis=0)
    std_error = np.std(pixel_errors, axis=0)

    # Calculate accuracy
    accuracy_x = (1 - mean_error[0]/orig_dimensions[0]) * 100
    accuracy_y = (1 - mean_error[1]/orig_dimensions[1]) * 100

    # Print results
    print("\nEvaluation Results:")
    print(f"Mean X Error (pixels): {mean_error[0]:.2f} ± {std_error[0]:.2f}")
    print(f"Mean Y Error (pixels): {mean_error[1]:.2f} ± {std_error[1]:.2f}")
    print(f"Accuracy X: {accuracy_x:.2f}%")
    print(f"Accuracy Y: {accuracy_y:.2f}%")

    # Create figure with subplots
    plt.figure(figsize=(20, 15))

    # 1. Scatter plot of real vs predicted gaze points
    plt.subplot(2, 2, 1)
    plt.scatter(y_true_pixels[:, 0], y_true_pixels[:, 1],
                alpha=0.5, label='Real Gaze', color='blue')
    plt.scatter(y_pred_pixels[:, 0], y_pred_pixels[:, 1],
                alpha=0.5, label='Predicted Gaze', color='red')
    plt.xlim(0, orig_dimensions[0])
    plt.ylim(0, orig_dimensions[1])
    plt.legend()
    plt.title('Real vs Predicted Gaze Points')
    plt.xlabel('X coordinate (pixels)')
    plt.ylabel('Y coordinate (pixels)')

    # 2. Heatmaps for real and predicted gazes
    plt.subplot(2, 2, 2)
    # Real gaze heatmap
    heatmap_real, xedges, yedges = np.histogram2d(
        y_true_pixels[:, 0], y_true_pixels[:, 1],
        bins=50, range=[[0, orig_dimensions[0]], [0, orig_dimensions[1]]]
    )
    plt.imshow(heatmap_real.T, origin='lower', cmap='hot',
               extent=[0, orig_dimensions[0], 0, orig_dimensions[1]])
    plt.colorbar(label='Frequency')
    plt.title('Real Gaze Heatmap')
    plt.xlabel('X coordinate (pixels)')
    plt.ylabel('Y coordinate (pixels)')

    # Predicted gaze heatmap
    plt.subplot(2, 2, 3)
    heatmap_pred, _, _ = np.histogram2d(
        y_pred_pixels[:, 0], y_pred_pixels[:, 1],
        bins=50, range=[[0, orig_dimensions[0]], [0, orig_dimensions[1]]]
    )
    plt.imshow(heatmap_pred.T, origin='lower', cmap='hot',
               extent=[0, orig_dimensions[0], 0, orig_dimensions[1]])
    plt.colorbar(label='Frequency')
    plt.title('Predicted Gaze Heatmap')
    plt.xlabel('X coordinate (pixels)')
    plt.ylabel('Y coordinate (pixels)')

    # 3. Normalized Error Histogram (0-1 range)
    plt.figure(figsize=(10, 5))

    # Calculate normalized Euclidean errors (0-1 range)
    screen_diagonal = np.sqrt(orig_dimensions[0]**2 + orig_dimensions[1]**2)
    errors = np.sqrt(np.sum((y_pred_pixels - y_true_pixels)**2, axis=1))
    normalized_errors = errors / screen_diagonal  # Now in 0-1 range

    # Plot histogram with smaller bins for higher resolution
    plt.hist(normalized_errors, bins=100, alpha=0.75, density=True)
    plt.title('Distribution of Normalized Prediction Errors')
    plt.xlabel('Error (proportion of screen diagonal)')
    plt.ylabel('Density')

    # Add mean and median lines
    mean_error = np.mean(normalized_errors)
    median_error = np.median(normalized_errors)
    plt.axvline(mean_error, color='red', linestyle='dashed', alpha=0.75,
                label=f'Mean: {mean_error:.3f}')
    plt.axvline(median_error, color='green', linestyle='dashed', alpha=0.75,
                label=f'Median: {median_error:.3f}')
    plt.legend()

    # 4. Sequential movement analysis
    plt.figure(figsize=(15, 5))

    # Calculate sequential distances for real gaze points
    real_distances = [euclidean(y_true_pixels[i], y_true_pixels[i+1])
                     for i in range(len(y_true_pixels)-1)]
    pred_distances = [euclidean(y_pred_pixels[i], y_pred_pixels[i+1])
                     for i in range(len(y_pred_pixels)-1)]

    # Plot sequential movement distances
    plt.subplot(1, 2, 1)
    plt.plot(real_distances, label='Real Gaze', alpha=0.7)
    plt.plot(pred_distances, label='Predicted Gaze', alpha=0.7)
    plt.title('Sequential Gaze Movement Distances')
    plt.xlabel('Frame Number')
    plt.ylabel('Distance (pixels)')
    plt.legend()

    # Distribution of movement distances
    plt.subplot(1, 2, 2)
    plt.hist(real_distances, bins=50, alpha=0.5, label='Real Gaze', color='blue')
    plt.hist(pred_distances, bins=50, alpha=0.5, label='Predicted Gaze', color='red')
    plt.title('Distribution of Movement Distances')
    plt.xlabel('Distance (pixels)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return accuracy_x, accuracy_y

eval_csv_path = 'drive/MyDrive/processed/processed_YCA.csv'
eval_video_path = 'drive/MyDrive/CNN/scenevideoYCA.mp4'
X_eval, y_eval, orig_dimensions = prepare_evaluation_data(eval_csv_path, eval_video_path)

# Evaluate using the existing model
evaluate_model(final_model, X_eval, y_eval, orig_dimensions)