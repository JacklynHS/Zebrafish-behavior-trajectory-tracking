import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_zebrafish_circular(
    csv_path,
    video_width,
    video_height,
    fps=30,
    center_radius_ratio=0.5,
    flip_y=False
):
    df = pd.read_csv(csv_path)

    if not {"CenterX", "CenterY"}.issubset(df.columns):
        raise ValueError("CSV file must contain CenterX and CenterY columns")

    print("Original coordinate range")
    print(f"X range: {df['CenterX'].min():.1f} to {df['CenterX'].max():.1f}")
    print(f"Y range: {df['CenterY'].min():.1f} to {df['CenterY'].max():.1f}")

    if flip_y:
        df["CenterY"] = video_height - df["CenterY"]
        print("Y axis has been flipped")

    center_x = video_width / 2
    center_y = video_height / 2
    arena_radius = min(video_width, video_height) / 2
    center_radius = arena_radius * center_radius_ratio

    df["DistanceToCenter"] = np.sqrt(
        (df["CenterX"] - center_x) ** 2 +
        (df["CenterY"] - center_y) ** 2
    )
    df["InCenter"] = df["DistanceToCenter"] <= center_radius

    frame_time = 1 / fps
    time_in_center = df["InCenter"].sum() * frame_time
    time_in_edge = (~df["InCenter"]).sum() * frame_time
    total_time = len(df) * frame_time

    print("Analysis results")
    print(f"Center radius ratio: {center_radius_ratio * 100:.0f} percent")
    print(f"Time in center: {time_in_center:.2f} seconds ({time_in_center / total_time:.1%})")
    print(f"Time at periphery: {time_in_edge:.2f} seconds ({time_in_edge / total_time:.1%})")
    print(f"Total time: {total_time:.2f} seconds")

    plt.figure(figsize=(8, 6))
    plt.plot(
        df["CenterX"],
        df["CenterY"],
        color="blue",
        alpha=0.7,
        linewidth=1,
        label="Trajectory"
    )
    plt.scatter(
        df["CenterX"].iloc[0],
        df["CenterY"].iloc[0],
        color="green",
        label="Starting point",
        zorder=3
    )
    plt.scatter(
        df["CenterX"].iloc[-1],
        df["CenterY"].iloc[-1],
        color="red",
        label="Ending point",
        zorder=3
    )

    central_circle = plt.Circle(
        (center_x, center_y),
        center_radius,
        color="orange",
        fill=False,
        linestyle="--",
        linewidth=2,
        label="Central area"
    )
    plt.gca().add_patch(central_circle)

    plt.title("Zebrafish Tracking in Circular Arena")
    plt.xlabel("X pixels")
    plt.ylabel("Y pixels")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.xlim(0, video_width)
    plt.ylim(0, video_height)

    plt.show()

    plt.figure(figsize=(6, 6))
    heatmap, xedges, yedges = np.histogram2d(
        df["CenterX"],
        df["CenterY"],
        bins=[50, 50],
        range=[[0, video_width], [0, video_height]]
    )
    sns.heatmap(
        np.flipud(heatmap.T),
        cmap="jet",
        cbar=True
    )
    plt.title("Zebrafish Heatmap in Circular Arena")
    plt.xlabel("X pixels")
    plt.ylabel("Y pixels")
    plt.axis("equal")
    plt.show()

    return {
        "time_in_center": time_in_center,
        "time_in_edge": time_in_edge,
        "total_time": total_time,
        "trajectory": df
    }


csv_path = input(
    "Please enter the path to the tracking CSV file:\n> "
)

result = analyze_zebrafish_circular(
    csv_path=csv_path,
    video_width=248,
    video_height=248,
    fps=200,
    center_radius_ratio=0.5,
    flip_y=True
)
