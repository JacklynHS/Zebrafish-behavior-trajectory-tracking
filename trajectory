# ============================================================================
# 1. Install compatible packages (use Colab default versions)
# ============================================================================

import subprocess
import sys

def install_requirements():
    """Install required packages using compatible versions"""
    print("Installing compatible packages...")

    # Use Colab-friendly versions to avoid conflicts
    packages = [
        "segment-anything",  # Original SAM, more stable
        "ultralytics",       # YOLOv8 as a backup tracker
    ]

    for package in packages:
        print(f"Installing {package}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"{package} installed successfully")
        else:
            print(f"{package} installation warning: {result.stderr[:100]}")

# Run installation
install_requirements()

# ============================================================================
# 2. Import packages
# ============================================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from PIL import Image
import csv
from datetime import datetime

# Check SAM availability
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    print("SAM imported successfully")
    SAM_AVAILABLE = True
except ImportError:
    print("SAM import failed, falling back to OpenCV tracking")
    SAM_AVAILABLE = False

print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# ============================================================================
# 3. Download SAM model (if available)
# ============================================================================

def download_sam_model():
    """Download SAM model weights"""
    if not SAM_AVAILABLE:
        return None

    model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    model_path = "/content/sam_vit_b_01ec64.pth"

    if not os.path.exists(model_path):
        print("Downloading SAM model...")
        try:
            import wget
            wget.download(model_url, model_path)
            print("\nModel download completed")
        except:
            os.system(f"curl -L {model_url} -o {model_path}")
            print("Model download completed")

    return model_path

model_path = download_sam_model()

# ============================================================================
# 4. Initialize models
# ============================================================================

def initialize_models():
    """Initialize segmentation and tracking models"""
    models = {}

    # SAM model
    if SAM_AVAILABLE and model_path and os.path.exists(model_path):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry["vit_b"](checkpoint=model_path)
            sam.to(device)
            models['sam_predictor'] = SamPredictor(sam)
            print("SAM model loaded successfully")
        except Exception as e:
            print(f"SAM loading failed: {e}")
            models['sam_predictor'] = None
    else:
        models['sam_predictor'] = None

    # OpenCV tracker (backup)
    try:
        models['opencv_tracker'] = cv2.TrackerCSRT_create()
        print("OpenCV tracker ready")
    except:
        try:
            models['opencv_tracker'] = cv2.legacy.TrackerCSRT_create()
            print("OpenCV legacy tracker ready")
        except:
            models['opencv_tracker'] = None
            print("OpenCV tracker unavailable")

    return models

models = initialize_models()

# ============================================================================
# 5. Mount Google Drive
# ============================================================================

print("\nMounting Google Drive...")
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully")
except Exception as e:
    print(f"Google Drive mount failed: {e}")

# ============================================================================
# 6. Set paths
# ============================================================================

VIDEO_PATH = input("Please type in the full path to your video file: ").strip()

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

print(f"Video found: {os.path.basename(VIDEO_PATH)}")

video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

OUTPUT_VIDEO = f"{video_name}_tracked.mp4"
OUTPUT_CSV = f"{video_name}_tracking.csv"

print(f"Output video: {OUTPUT_VIDEO}")
print(f"Output CSV: {OUTPUT_CSV}")

# ============================================================================
# 7. Get video information
# ============================================================================

def get_video_info(video_path):
    """Retrieve basic video information"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    cap.release()

    print("Video information:")
    print(f" Resolution: {width}x{height}")
    print(f" FPS: {fps}")
    print(f" Total frames: {frame_count}")
    print(f" Duration: {duration:.1f} seconds")

    return {
        'fps': fps,
        'width': width,
        'height': height,
        'frame_count': frame_count,
        'duration': duration
    }

if os.path.exists(VIDEO_PATH):
    video_info = get_video_info(VIDEO_PATH)

# ============================================================================
# 8. Install additional packages
# ============================================================================

def install_packages():
    """Install additional Colab-compatible packages"""
    packages = [
        "imageio[ffmpeg]",
        "scikit-image",
        "matplotlib",
        "pandas"
    ]

    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} installed successfully")
        except:
            print(f"{package} installation may have issues")

install_packages()

# ============================================================================
# 9. SAM-based tracking function
# ============================================================================

def track_with_sam_from_frame300(initial_frame, initial_bbox):
    """Track zebrafish using SAM starting from frame 300"""
    print("Starting SAM tracking from frame 300...")

    sam_predictor = models.get('sam_predictor')
    if sam_predictor is None:
        print("SAM model unavailable")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height}, {fps} FPS")
    print(f"Tracking remaining {frame_count - 300} frames")

    output_video = "/content/zebrafish_tracked_from_300.mp4"
    output_csv = "/content/zebrafish_tracking_data_from_300.csv"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    with open(output_csv, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ['Frame', 'Time', 'X', 'Y', 'Width', 'Height', 'CenterX', 'CenterY']
        )

        current_bbox = initial_bbox
        sam_update_interval = 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, 300)

        for frame_idx in tqdm(range(300, frame_count), desc="Tracking progress"):
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / fps

            if frame_idx == 300 or (frame_idx - 300) % sam_update_interval == 0:
                sam_predictor.set_image(frame)
                x, y, w, h = current_bbox
                cx, cy = x + w // 2, y + h // 2

                masks, scores, _ = sam_predictor.predict(
                    point_coords=np.array([[cx, cy]]),
                    point_labels=np.array([1]),
                    box=np.array([x, y, x + w, y + h]),
                    multimask_output=True
                )

                best_mask = masks[np.argmax(scores)]
                ys, xs = np.where(best_mask)

                if len(xs) > 0:
                    current_bbox = (
                        int(xs.min()), int(ys.min()),
                        int(xs.max() - xs.min()),
                        int(ys.max() - ys.min())
                    )

            x, y, w, h = current_bbox
            cx, cy = x + w // 2, y + h // 2
            csv_writer.writerow([frame_idx, f"{current_time:.2f}", x, y, w, h, cx, cy])

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            out.write(frame)

    cap.release()
    out.release()
    print("SAM tracking completed")
