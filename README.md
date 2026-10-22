# Zebrafish Trajectory and Behavioral Analysis

Python tools for tracking zebrafish movement from video recordings and analyzing spatial behavior.

This project includes zebrafish trajectory tracking, spatial occupancy heatmaps, and behavioral analysis.

## Repository Structure

```text
.
├── README.md
├── trajectory.py
├── heatmap.py
└── behavior_analysis.py
```

## Files

### `trajectory.py`

Tracks zebrafish movement from video recordings using the Segment Anything Model (SAM).

Outputs include:

- Frame-by-frame zebrafish position
- Bounding box coordinates
- Center X/Y coordinates
- Tracking CSV file
- Annotated tracking video

### `heatmap.py`

Analyzes zebrafish spatial behavior using the tracking CSV file.

Generates:

- Swimming trajectory
- Spatial occupancy heatmap
- Time spent in the center
- Time spent in the periphery

For a circular arena, the center region can be defined using:

```python
center_radius_ratio = 0.5
```

A value of `0.5` means that the center radius is 50% of the arena radius, corresponding to 25% of the total circular arena area.

### `analysis_of_depression.py`

Performs additional behavioral analysis using zebrafish tracking data.

## Requirements

Install the required Python packages:

```bash
pip install segment-anything ultralytics pandas numpy matplotlib seaborn opencv-python scikit-image imageio[ffmpeg]
```

The scripts are designed primarily for **Google Colab**.

GPU acceleration is recommended for SAM-based tracking.

## Usage

### 1. Zebrafish Tracking

Run:

```bash
python trajectory.py
```

Enter the path to the zebrafish video when prompted.

The tracking CSV contains:

```text
Frame, Time, X, Y, Width, Height, CenterX, CenterY
```

### 2. Trajectory and Heatmap Analysis

Run:

```bash
python heatmap.py
```

Enter the path to the tracking CSV file when prompted.

Configure the video parameters according to your recording:

```python
video_width = 248
video_height = 248
fps = 200
```

Example:

```python
result = analyze_zebrafish_circular(
    csv_path=csv_path,
    video_width=248,
    video_height=248,
    fps=200,
    center_radius_ratio=0.5,
    flip_y=True
)
```

## Output

The workflow generates:

- Zebrafish trajectory plots
- Spatial occupancy heatmaps
- Center/periphery dwell time
- Tracking CSV files
- Annotated tracking videos

## Notes

Tracking accuracy should be visually inspected before behavioral analysis.

The circular-arena analysis assumes that the arena is approximately

<img width="1440" height="577" alt="Trajectory analysis results" src="https://github.com/user-attachments/assets/53892520-3f2a-4553-b3a3-33d0044ab749" />
