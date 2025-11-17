import cv2
import numpy as np
import pandas as pd
import math
from collections import defaultdict
from ultralytics import YOLO

# ---------------- CONFIG ------------------------------------------------
POSE_MODEL   = 'yolov8l-pose.pt'
FULL_IMGSZ   = 960
CONF_THR     = 0.35
AFFINITY_K   = 0.12
THICK        = 2

# I/O and time window ----------------------------------------------------
VIDEO_PATH   = 'drive/MyDrive/CNN/scenevideoYCA.mp4'
CSV_PATH     = 'drive/MyDrive/processed/processed_YCA.csv'
START_TIME   = '13:47'   # MM:SS  (inclusive)
END_TIME     = '14:22'   # MM:SS  (exclusive)
FPS_ASSUMED  = 24.95     

# Output video path
OUTPUT_VIDEO = 'processed_video.mp4'

# ---------------- Keypoint / Region mappings (COCO-17) -----------------
BODY_PARTS = {
    'nose':0,'left_eye':1,'right_eye':2,'left_ear':3,'right_ear':4,
    'left_shoulder':5,'right_shoulder':6,'left_elbow':7,'right_elbow':8,
    'left_wrist':9,'right_wrist':10,'left_hip':11,'right_hip':12,
    'left_knee':13,'right_knee':14,'left_ankle':15,'right_ankle':16
}

# Simple skeleton for drawing (pairs of indices)
SKEL = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (11,12),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)
]

# Aggregation to high-level regions for final statistics
BODY_REGIONS = {
    'Head'        : ['nose','left_eye','right_eye','left_ear','right_ear'],
    'Trunk'       : ['left_shoulder','right_shoulder','left_hip','right_hip', 'Trunk'],  # include 'Trunk' fallback label
    'Legs'        : ['left_knee','right_knee'],
    'Feet'        : ['left_ankle','right_ankle'],
    'Ground'      : [],
    'Environment' : []
}

# --------------------------- Helper Functions ---------------------------
def time_to_frame(time_str, fps):
    """Convert a 'MM:SS' time string to an absolute frame index using fps."""
    minutes, seconds = map(int, time_str.split(':'))
    return int((minutes * 60 + seconds) * fps)

def robust_read_csv_three_cols(csv_path):
    """
    Read CSV that has at least three columns: frame_index, gaze_x, gaze_y.
    Works whether the CSV has headers or not. Returns numpy arrays (int, float, float).
    """
    try:
        df = pd.read_csv(csv_path)
        f0 = pd.to_numeric(df.iloc[:,0], errors='coerce')
        f1 = pd.to_numeric(df.iloc[:,1], errors='coerce')
        f2 = pd.to_numeric(df.iloc[:,2], errors='coerce')
        if f0.notna().all() and f1.notna().all() and f2.notna().all():
            return f0.astype(int).to_numpy(), f1.astype(float).to_numpy(), f2.astype(float).to_numpy()
    except Exception:
        pass

    df = pd.read_csv(csv_path, header=None)
    f0 = pd.to_numeric(df.iloc[:,0], errors='coerce')
    f1 = pd.to_numeric(df.iloc[:,1], errors='coerce')
    f2 = pd.to_numeric(df.iloc[:,2], errors='coerce')
    if not (f0.notna().all() and f1.notna().all() and f2.notna().all()):
        raise ValueError("CSV must have three numeric columns: frame_index, gaze_x, gaze_y.")
    return f0.astype(int).to_numpy(), f1.astype(float).to_numpy(), f2.astype(float).to_numpy()

def load_subset(csv_path, video_path, start_time, end_time, fps_assumed):
    """
    Load a subset of frames and corresponding gaze points for [start_time, end_time).
    Returns: frames (list of BGR), gazes (list of (x,y)), width, height
    """
    idx, gx, gy = robust_read_csv_three_cols(csv_path)
    s_f = time_to_frame(start_time, fps_assumed)
    e_f = time_to_frame(end_time,   fps_assumed)

    sel_mask = (idx >= s_f) & (idx < e_f)
    idx_sel = idx[sel_mask]
    gx_sel  = gx[sel_mask]
    gy_sel  = gy[sel_mask]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frames, gazes = [], []
    for fi, xg, yg in zip(idx_sel, gx_sel, gy_sel):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
            gazes.append((float(xg), float(yg)))
    cap.release()
    return frames, gazes, W, H

def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def is_pose_vertical(kp_xy):
    """
    Shoulders above hips heuristic.
    kp_xy: (17,2) array of ints for a single person.
    Returns True/False; robust to missing by requiring both shoulders and hips valid (>0).
    """
    ls, rs = BODY_PARTS['left_shoulder'], BODY_PARTS['right_shoulder']
    lh, rh = BODY_PARTS['left_hip'], BODY_PARTS['right_hip']
    pts = kp_xy

    def ok(i): return not (pts[i][0] == 0 and pts[i][1] == 0)

    if ok(ls) and ok(rs) and ok(lh) and ok(rh):
        shoulder_y = (pts[ls][1] + pts[rs][1]) / 2.0
        hip_y      = (pts[lh][1] + pts[rh][1]) / 2.0
        return shoulder_y < hip_y
    return False

def body_part_at_gaze(gx, gy, pts, valid_mask, radius):
    """
    Return the closest body-part name whose keypoint is within 'radius'
    of (gx,gy). Uses only keypoints where valid_mask==True.
    """
    inside = []
    for name, idx in BODY_PARTS.items():
        if idx < len(pts) and valid_mask[idx]:
            px, py = pts[idx]
            d = math.hypot(gx - px, gy - py)
            if d <= radius:
                inside.append((name, d))
    if inside:
        return min(inside, key=lambda x: x[1])[0]
    return None

def trunk_center(pts):
    """Mean of left/right shoulders and hips (returns (x,y))."""
    ls, rs = BODY_PARTS['left_shoulder'], BODY_PARTS['right_shoulder']
    lh, rh = BODY_PARTS['left_hip'], BODY_PARTS['right_hip']
    sub = np.array([pts[ls], pts[rs], pts[lh], pts[rh]], dtype=np.float32)
    return np.mean(sub, axis=0)

# --------------------------- Drawing -----------------------------------
def draw_skeleton(img, pts, valid_mask, color_lines=(255,0,0), color_joints=(0,255,255), thick=2):
    out = img
    for a, b in SKEL:
        if a < len(pts) and b < len(pts) and valid_mask[a] and valid_mask[b]:
            cv2.line(out, tuple(map(int, pts[a])), tuple(map(int, pts[b])), color_lines, thick)
    for i, (x, y) in enumerate(pts):
        if i < len(valid_mask) and valid_mask[i]:
            cv2.circle(out, (int(x), int(y)), 4, color_joints, -1)
    return out

def draw_affinity_circles(img, pts, valid_mask, radius, color=(0,255,0), thick=2):
    out = img
    for i, (x, y) in enumerate(pts):
        if i < len(valid_mask) and valid_mask[i]:
            cv2.circle(out, (int(x), int(y)), int(radius), color, thick)
    return out

# ------------------------------ Main -----------------------------------
if __name__ == '__main__':
    # Load YOLOv8-Pose model
    pose_model = YOLO(POSE_MODEL)

    # Load frames + gazes for the selected time window
    frames, gazes, W, H = load_subset(CSV_PATH, VIDEO_PATH, START_TIME, END_TIME, FPS_ASSUMED)
    if len(frames) == 0:
        raise RuntimeError("No frames loaded for the specified time window. Check times / CSV / FPS.")

    # Prepare output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS_ASSUMED, (W, H))

    # Counters
    body_region_count = defaultdict(int)
    total_observations = 0

    # Process each frame (FULL FRAME — no vertical slice filter)
    for frame, (gx, gy) in zip(frames, gazes):
        # Run pose on the ORIGINAL frame for best detection
        res = pose_model(frame, imgsz=FULL_IMGSZ if FULL_IMGSZ else None, verbose=False)[0]

        # Prepare the drawable image
        image = frame.copy()

        # Safety for NaN gazes
        if gx is None or gy is None or np.isnan(gx) or np.isnan(gy):
            out_writer.write(image)
            continue

        # Draw gaze cross
        cv2.drawMarker(image, (int(gx), int(gy)), (0, 0, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        closest_label = 'invalid'
        closest_dist  = float('inf')

        # If we have keypoints for detected persons
        if hasattr(res, 'keypoints') and res.keypoints is not None and res.keypoints.xy is not None:
            kp_xy   = res.keypoints.xy
            boxes   = res.boxes.xyxy if res.boxes is not None else None
            kp_conf = res.keypoints.conf if res.keypoints.conf is not None else None

            if kp_xy is not None and kp_xy.numel():
                kp_xy   = kp_xy.cpu().numpy().astype(int)           # (N,17,2)
                boxes   = boxes.cpu().numpy().astype(int) if boxes is not None else np.zeros((0,4), dtype=int)
                kp_conf = kp_conf.cpu().numpy() if kp_conf is not None and kp_conf.numel() else np.zeros(kp_xy.shape[:2], dtype=float)

                # Iterate over persons
                for pts, confs, box in zip(kp_xy, kp_conf, boxes):
                    x1, y1, x2, y2 = box

                    # Person height for dynamic affinity radius
                    person_h = max(1, (y2 - y1))
                    radius   = int(AFFINITY_K * person_h)

                    # Valid mask = confidence >= CONF_THR and point inside person box
                    inside = (pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= y1) & (pts[:, 1] <= y2)
                    valid  = (confs >= CONF_THR) & inside

                    # Draw skeleton and affinity regions
                    image = draw_skeleton(image, pts, valid, color_lines=(255, 255, 0), color_joints=(0, 255, 0), thick=THICK)
                    image = draw_affinity_circles(image, pts, valid, radius, color=(0, 255, 0), thick=THICK)

                    # Pose vertical heuristic: if vertical, allow trunk fallback
                    vertical = is_pose_vertical(pts)

                    # 1) try keypoint circles
                    part = body_part_at_gaze(gx, gy, pts, valid, radius)
                    if part is not None:
                        bx, by = pts[BODY_PARTS[part]]
                        dist   = euclidean_distance((gx, gy), (bx, by))
                        if dist < closest_dist:
                            closest_dist  = dist
                            closest_label = part

                    # 2) trunk center fallback (if vertical and within radius)
                    if vertical:
                        tcx, tcy = trunk_center(pts)
                        tdist = euclidean_distance((gx, gy), (tcx, tcy))
                        if tdist <= radius and tdist < closest_dist:
                            closest_dist  = tdist
                            closest_label = 'Trunk'

        # If no body part found, assign Ground/Environment by gaze_y
        if closest_label == 'invalid':
            if gy > H * 0.5:
                closest_label = 'Ground'
            else:
                closest_label = 'Environment'

        # Count and annotate
        body_region_count[closest_label] += 1
        total_observations += 1

        cv2.putText(image, f'{closest_label}', (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        # Write annotated frame
        out_writer.write(image)

    # Cleanup writer
    out_writer.release()
    cv2.destroyAllWindows()

    # --------------------------- Final Statistics ---------------------------
    if total_observations > 0:
        # Convert fine-grained counts to percentages
        part_percent = {k: (v / total_observations) * 100.0 for k, v in body_region_count.items()}
    else:
        part_percent = {}

    # Aggregate to requested regions
    aggregated = {
        'Head'        : sum(part_percent.get(p, 0.0) for p in BODY_REGIONS['Head']),
        'Trunk'       : sum(part_percent.get(p, 0.0) for p in BODY_REGIONS['Trunk']),
        'Legs'        : sum(part_percent.get(p, 0.0) for p in BODY_REGIONS['Legs']),
        'Feet'        : sum(part_percent.get(p, 0.0) for p in BODY_REGIONS['Feet']),
        'Ground'      : part_percent.get('Ground', 0.0),
        'Environment' : part_percent.get('Environment', 0.0)
    }

    print("\nPercentage of time spent looking at different body regions:")
    for region in ['Head', 'Trunk', 'Legs', 'Feet', 'Ground', 'Environment']:
        print(f"{region}: {aggregated[region]:.2f}%")
