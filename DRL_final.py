import socket
import cv2
import numpy as np
import struct
import threading
import os
import random
from collections import deque
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, Concatenate, ReLU, Rescaling
from tensorflow.keras import backend as K
import tensorflow as tf
import time

reset_lock = threading.Lock()

# ---------------------------
# Custom RMSE loss for gaze model
# ---------------------------
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# ---------------------------
# Load the gaze prediction model (inference-only)
# ---------------------------
GAZE_MODEL_PATH = 'CCA_model.h5'
if not os.path.exists(GAZE_MODEL_PATH):
    raise FileNotFoundError(f"Gaze model file '{GAZE_MODEL_PATH}' not found.")
gaze_model = load_model(GAZE_MODEL_PATH, custom_objects={'root_mean_squared_error': root_mean_squared_error})

FOVEATED_INPUT_SHAPE = (270, 480, 3)  # H, W, C for policy visual branch

def create_policy_network(input_shape, action_space):
    """
    Creates a two-branch DQN-style policy:
      - state_input: telemetry vector of length `input_shape` (excluding gaze)
      - image_input: foveated RGB image of shape FOVEATED_INPUT_SHAPE

    Output:
      - action_output: Q-values for `action_space` discrete actions (linear activation)
    """
    # Telemetry branch (numeric state vector)
    state_input = Input(shape=(input_shape,), name='state_input')
    t = Dense(32, activation='relu', name='tel_dense1')(state_input)
    t = Dense(32, activation='relu', name='tel_dense2')(t)

    # Visual branch (foveated image)
    image_input = Input(shape=FOVEATED_INPUT_SHAPE, name='image_input')
    x = Rescaling(1.0 / 255.0, name='rescale_0_1')(image_input)
    x = Conv2D(32, kernel_size=8, strides=4, padding='same', name='conv1')(x)
    x = ReLU(name='relu1')(x)
    x = Conv2D(64, kernel_size=4, strides=2, padding='same', name='conv2')(x)
    x = ReLU(name='relu2')(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same', name='conv3')(x)
    x = ReLU(name='relu3')(x)
    x = GlobalAveragePooling2D(name='gap')(x)
    v = Dense(128, activation='relu', name='vision_embed')(x)

    # Fusion + Q head
    fused = Concatenate(name='fusion')([v, t])
    z = Dense(128, activation='relu', name='fusion_dense1')(fused)
    z = Dense(64, activation='relu', name='fusion_dense2')(z)
    action_output = Dense(action_space, activation='linear', name='action_output')(z)

    model = Model(inputs=[state_input, image_input], outputs=action_output)
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------------------------
# Parameters
# ---------------------------
STATE_SIZE = 5

ACTION_SPACE = 4  # {1: Left 15°, 2: Right 15°, 3: Left 45°, 4: Right 45°}

policy_model = create_policy_network(STATE_SIZE, ACTION_SPACE)
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
gamma = 0.99
batch_size = 32
replay_memory = deque(maxlen=10000)

POLICY_MODEL_PATH = 'policy_model.h5'
if os.path.exists(POLICY_MODEL_PATH):
    try:
        policy_model.load_weights(POLICY_MODEL_PATH)
        print("Loaded pre-trained policy model.")
    except Exception as e:
        print(f"Found '{POLICY_MODEL_PATH}' but could not load weights due to architecture mismatch or shape error: {e}")
        print("Proceeding with randomly initialized policy model.")
else:
    print("No pre-trained policy model found. Using randomly initialized model.")

# ---------------------------
# Experience replay functions
# ---------------------------
def store_transition(state, action, reward, next_state, done):
    """
    Store transitions where:
      state      = (state_vec, state_img)
      next_state = (next_state_vec, next_state_img)
      state_vec shape: (STATE_SIZE,)
      state_img shape: (H, W, C) i.e., FOVEATED_INPUT_SHAPE
    """
    replay_memory.append((state, action, reward, next_state, done))

def sample_batch():
    return random.sample(replay_memory, batch_size)

def train_step():
    if len(replay_memory) < batch_size:
        return

    batch = sample_batch()
    states, actions, rewards, next_states, dones = zip(*batch)

    # Unpack telemetry vectors and images
    state_vecs = np.array([s[0] for s in states], dtype=np.float32)                  # (B, STATE_SIZE)
    state_imgs = np.array([s[1] for s in states], dtype=np.uint8)                   # (B, H, W, C)
    next_state_vecs = np.array([ns[0] for ns in next_states], dtype=np.float32)     # (B, STATE_SIZE)
    next_state_imgs = np.array([ns[1] for ns in next_states], dtype=np.uint8)       # (B, H, W, C)

    # Predict current and next Q-values
    q_values = policy_model.predict([state_vecs, state_imgs], verbose=0)
    next_q_values = policy_model.predict([next_state_vecs, next_state_imgs], verbose=0)

    # Update targets
    for i in range(batch_size):
        a_idx = actions[i] - 1  # actions are 1..ACTION_SPACE
        if dones[i]:
            q_values[i][a_idx] = rewards[i]
        else:
            q_values[i][a_idx] = rewards[i] + gamma * float(np.max(next_q_values[i], axis=-1))

    policy_model.train_on_batch([state_vecs, state_imgs], q_values)

# ---------------------------
# Image helpers
# ---------------------------
def preprocess_image(image_data, target_width, target_height):
    """
    Decode a JPEG-encoded image byte array and resize for the gaze model (480x270).
    Returns array of shape (1, H, W, C) for the gaze model.
    """
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return None
    image = cv2.resize(image, (target_width, target_height))
    image = np.expand_dims(image, axis=0)
    return image

def apply_foveal_effect(image, gaze_coordinates, radius=100, blend_width=50):
    """
    Apply a foveation effect centered at gaze_coordinates on the original BGR image.
    Returns an foveated image with same size as input.
    """
    height, width, _ = image.shape
    # Scale gaze coordinates from (0..480, 0..270) up to actual image size
    x = int(gaze_coordinates[0] * (width / 480.0))
    y = int(gaze_coordinates[1] * (height / 270.0))

    blurred_image = cv2.GaussianBlur(image, (21, 21), 0)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, (255), thickness=-1)
    blurred_mask = cv2.GaussianBlur(mask, (2 * blend_width + 1, 2 * blend_width + 1), blend_width)
    normalized_mask = blurred_mask / 255.0
    foveal_image = (image * normalized_mask[..., None] + blurred_image * (1 - normalized_mask[..., None])).astype(np.uint8)
    return foveal_image

def prepare_policy_image(foveal_image):
    """
    Resize foveated image to the policy network input size FOVEATED_INPUT_SHAPE.
    Returns array of shape (H, W, C) without batch dimension.
    """
    h, w, c = FOVEATED_INPUT_SHAPE
    resized = cv2.resize(foveal_image, (w, h))  # cv2 uses (width, height)
    return resized

# ---------------------------
# Socket helpers
# ---------------------------
def receive_all(sock, count):
    buffer = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buffer += newbuf
        count -= len(newbuf)
    return buffer

def send_data(conn, data):
    conn.sendall(data)

# ---------------------------
# Epsilon-greedy
# ---------------------------
def drl_decision(state):
    """
    state: (state_vec, state_img)
      - state_vec: shape (STATE_SIZE,)
      - state_img: shape (H, W, C)
    """
    global epsilon
    if random.random() < epsilon:
        action_code = random.randint(1, ACTION_SPACE)
    else:
        state_vec, state_img = state
        state_vec_input = np.asarray(state_vec, dtype=np.float32).reshape(1, -1)
        state_img_input = np.asarray(state_img, dtype=np.uint8)[np.newaxis, ...]
        action_scores = policy_model.predict([state_vec_input, state_img_input], verbose=0)[0]
        action_code = int(np.argmax(action_scores)) + 1
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    return action_code

# ---------------------------
# Global visualization state
# ---------------------------
latest_foveal_image = None
foveal_lock = threading.Lock()

# ---------------------------
# Episode tracking
# ---------------------------
current_episode_reward = 0.0
episode_rewards = []
episode_count = 0

# ---------------------------
# Logging
# ---------------------------
REWARD_LOG_FILE = 'episode_rewards.log'
def log_reward(reward, episode_number):
    with open(REWARD_LOG_FILE, 'a') as f:
        f.write(f"Episode {episode_number}: Total Reward = {reward}\n")

# ---------------------------
# Heuristic thresholds to compute collision_flag
# ---------------------------
ACCEL_SMOOTHING_MIN_DT = 1e-3   # avoid div-by-zero
ACCEL_COLLISION_DECEL = -1.5    # if decel less than this (units/s^2), suspect collision
VEL_NEAR_ZERO = 0.2             # velocity under this considered near-stop
DRIFT_THRESHOLD = 0.5           # if |deltaX| or |deltaZ| exceeds, suspect drift/impact

def handle_client(conn, addr):
    """
    Main loop:
      - Receive frame + telemetry (velocity, deltaX, deltaZ)
      - Run gaze model to get (gx, gy)
      - Build foveated image and resize for policy input
      - Build telemetry state vector [velocity, acceleration, collision_flag, deltaX, deltaZ]
      - Choose action via epsilon-greedy and send
      - Receive reward & done
      - Store transition in replay buffer and train
    """
    global latest_foveal_image, current_episode_reward, episode_count
    print(f"Connected by {addr}")

    # Previous step bookkeeping for acceleration & replay
    prev_velocity = None
    prev_time = None

    prev_state = None          # (state_vec, state_img)
    prev_action = None
    last_reward = None
    prev_done_flag = False

    done = False
    reset_pending = False
    host = '127.0.0.1'
    port = 8188

    try:
        while True:
            if reset_pending:
                conn.close()
                time.sleep(0.2)
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.connect((host, port))
                # reset step trackers
                prev_velocity = None
                prev_time = None
                prev_state = None
                prev_action = None
                last_reward = None
                prev_done_flag = False
                reset_pending = False
                continue

            # Receive image length
            length_prefix = receive_all(conn, 4)
            if not length_prefix:
                print("Connection closed by client.")
                break
            image_length = struct.unpack('!I', length_prefix)[0]

            # Receive image bytes
            image_data = receive_all(conn, image_length)
            if not image_data:
                print("No image data received. Closing connection.")
                break

            # Receive telemetry
            speed_bytes = receive_all(conn, 4)
            if not speed_bytes:
                break
            velocity = struct.unpack('!f', speed_bytes)[0]

            deltaX_bytes = receive_all(conn, 4)
            if not deltaX_bytes:
                break
            deltaX = struct.unpack('!f', deltaX_bytes)[0]

            deltaZ_bytes = receive_all(conn, 4)
            if not deltaZ_bytes:
                break
            deltaZ = struct.unpack('!f', deltaZ_bytes)[0]

            # 1) Run gaze model to get gaze center on 480x270 frame
            processed_image = preprocess_image(image_data, 480, 270)  # shape (1, 270, 480, 3)
            if processed_image is None:
                print("Failed to preprocess image for gaze model.")
                continue

            predicted_gaze = gaze_model.predict(processed_image, verbose=0)[0]  # [gx, gy] in 480x270 coords

            # 2) Build foveated image from original full-res frame
            original_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            if original_image is None:
                print("Failed to decode original image for foveation.")
                continue

            foveal_image = apply_foveal_effect(original_image, predicted_gaze)

            with foveal_lock:
                latest_foveal_image = foveal_image

            # 3) Prepare policy image (H,W,C)
            policy_img = prepare_policy_image(foveal_image)  # (270,480,3)

            # 4) Compute acceleration and collision_flag (no placeholders)
            now_t = time.time()
            if prev_velocity is None or prev_time is None:
                acceleration = 0.0
            else:
                dt = max(ACCEL_SMOOTHING_MIN_DT, now_t - prev_time)
                acceleration = (velocity - prev_velocity) / dt

            # Simple collision heuristic: sudden strong deceleration OR near-stop with significant drift
            collision_flag = 1.0 if (
                (acceleration < ACCEL_COLLISION_DECEL) or
                (abs(velocity) < VEL_NEAR_ZERO and (abs(deltaX) > DRIFT_THRESHOLD or abs(deltaZ) > DRIFT_THRESHOLD))
            ) else 0.0

            prev_velocity = velocity
            prev_time = now_t

            # 5) Build telemetry state vector (EXCLUDING gaze): [v, a, coll, dX, dZ]
            state_vec = [
                float(velocity),
                float(acceleration),
                float(collision_flag),
                float(deltaX),
                float(deltaZ),
            ]
            state_tuple = (np.asarray(state_vec, dtype=np.float32), np.asarray(policy_img, dtype=np.uint8))

            # 6) If we have a previous state-action and reward, finalize that transition now with current state as next_state
            if prev_state is not None and prev_action is not None and last_reward is not None:
                store_transition(prev_state, prev_action, last_reward, state_tuple, prev_done_flag)
                train_step()

            # 7) Choose and send action for the CURRENT state
            last_action = drl_decision(state_tuple)
            packed_action = struct.pack('!I', last_action)
            send_data(conn, packed_action)

            # 8) Receive reward for the action we just sent
            reward_bytes = receive_all(conn, 4)
            if not reward_bytes:
                print("No reward received. Closing connection.")
                break
            reward = struct.unpack('!f', reward_bytes)[0]
            current_episode_reward += reward

            # 9) Receive done flag
            done_bytes = receive_all(conn, 4)
            if not done_bytes:
                print("No done flag received. Closing connection.")
                break
            done_flag = struct.unpack('!f', done_bytes)[0]
            done = bool(done_flag == 1.0)

            # 10) If episode terminates now, store terminal transition immediately
            if done:
                # Terminal transition: next_state won't be used by target (done=True)
                store_transition(state_tuple, last_action, reward, state_tuple, True)
                train_step()

                episode_rewards.append(current_episode_reward)
                episode_count += 1
                print(f"Episode {episode_count} ended. Total Reward: {current_episode_reward}")
                log_reward(current_episode_reward, episode_count)

                # Reset for next episode
                current_episode_reward = 0.0
                prev_state = None
                prev_action = None
                last_reward = None
                prev_done_flag = False
                reset_pending = True
                continue

            # 11) Otherwise keep current as "previous" for the next iteration
            prev_state = state_tuple
            prev_action = last_action
            last_reward = reward
            prev_done_flag = False

    except Exception as e:
        print(f"Exception occurred: {e}")
    finally:
        try:
            conn.shutdown(socket.SHUT_WR)
            conn.close()
        except:
            pass
        print(f"Connection with {addr} closed.")

# ---------------------------
# Misc helpers
# ---------------------------
def yield_thread():
    time.sleep(0.05)

def display_foveal_images():
    global latest_foveal_image
    while True:
        if latest_foveal_image is not None:
            cv2.imshow("Foveal Image", latest_foveal_image)
        else:
            blank_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.imshow("Foveal Image", blank_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Foveal image display window closed by user.")
            break
    cv2.destroyAllWindows()
    os._exit(0)

def server():
    HOST = '127.0.0.1'
    PORT = 8188

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(5)
    print(f"Server listening on {HOST}:{PORT}")

    # Start the foveal display in another thread
    display_thread = threading.Thread(target=display_foveal_images, daemon=True)
    display_thread.start()

    try:
        while True:
            conn, addr = s.accept()
            client_thread = threading.Thread(target=handle_client, args=(conn, addr))
            client_thread.start()
    except KeyboardInterrupt:
        print("Server shutting down.")
    finally:
        s.close()
        policy_model.save_weights(POLICY_MODEL_PATH)
        print(f"Policy model weights saved to {POLICY_MODEL_PATH}")

if __name__ == "__main__":
    server()