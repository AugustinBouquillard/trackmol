# -*- coding: utf-8 -*-
"""
This script simulates and visualizes various types of random walks in 2D space.
It includes functions for generating different types of random walks, creating
animations, and saving data to CSV files. The script demonstrates how to use
these functions to generate walks with and without boundaries, save them to
files, and create animations.
"""
import numpy as np
from fbm import FBM
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2
import tifffile as tiff
import pandas as pd


# 1. Brownian Motion (2D)
def bm_2d(steps=1000, dt=1.0, boundaries=None, start_position=None):
    """
    Generates a 2D random walk with optional boundaries.

    Parameters:
    -----------
    steps : int
        Number of steps.
    dt : float
        Time increment.
    boundaries : tuple or None
        (xmin, xmax, ymin, ymax) for boundaries.
    start_position : tuple or None
        Initial position (x_start, y_start).

    Returns:
    --------
    tuple of arrays
        (position_x, position_y) coordinates.
    """
    increments_x = np.random.normal(loc=0, scale=np.sqrt(dt), size=steps) * 10
    increments_y = np.random.normal(loc=0, scale=np.sqrt(dt), size=steps) * 10
    position_x = np.zeros(steps)
    position_y = np.zeros(steps)

    if start_position is not None:
        position_x[0], position_y[0] = start_position
    else:
        position_x[0], position_y[0] = 0, 0

    for i in range(1, steps):
        position_x[i] = position_x[i - 1] + increments_x[i]
        position_y[i] = position_y[i - 1] + increments_y[i]

        if boundaries:
            xmin, xmax, ymin, ymax = boundaries
            # Vérification des collisions avec les limites en x
            if position_x[i] < xmin:
                position_x[i] = 2 * xmin - position_x[i]
                increments_x[i] = -increments_x[i]
            elif position_x[i] > xmax:
                position_x[i] = 2 * xmax - position_x[i]
                increments_x[i] = -increments_x[i]

            # Vérification des collisions avec les limites en y
            if position_y[i] < ymin:
                position_y[i] = 2 * ymin - position_y[i]
                increments_y[i] = -increments_y[i]
            elif position_y[i] > ymax:
                position_y[i] = 2 * ymax - position_y[i]
                increments_y[i] = -increments_y[i]

    return position_x, position_y


# # 2. Fractional Brownian Motion (2D)
def fbm_2d(steps=1000, hurst=0.5, boundaries=None, start_position=None):
    """
    Generates a 2D Fractional Brownian motion using the fbm library with optional reflective boundaries.

    Parameters:
    -----------
    steps : int
        Number of steps in the motion.
    hurst : float
        Hurst exponent (0 < H < 1) for the fractional Brownian motion.
    boundaries : tuple or None
        (xmin, xmax, ymin, ymax) for reflective boundaries, or None for no boundaries.
    start_position : tuple or None
        Initial position (x_start, y_start) or None for starting at (0,0).

    Returns:
    --------
    tuple of arrays
        (position_x, position_y) containing the trajectory coordinates.
    """
    f_x = FBM(n=steps, hurst=hurst, length=1, method='daviesharte')
    f_y = FBM(n=steps, hurst=hurst, length=1, method='daviesharte')
    position_x = f_x.fbm()
    position_y = f_y.fbm()
    if start_position is not None:
        position_x += start_position[0]
        position_y += start_position[1]
    if boundaries:
        xmin, xmax, ymin, ymax = boundaries
        for i in range(len(position_x)):
            # Check for collisions with x boundaries
            if position_x[i] < xmin:
                position_x[i] = 2 * xmin - position_x[i]  # Reflect back into bounds
            elif position_x[i] > xmax:
                position_x[i] = 2 * xmax - position_x[i]
            # Check for collisions with y boundaries
            if position_y[i] < ymin:
                position_y[i] = 2 * ymin - position_y[i]
            elif position_y[i] > ymax:
                position_y[i] = 2 * ymax - position_y[i]

    return position_x, position_y


# 3. Continuous Time Random Walks (CTRW) (2D)
def ctrw_2d(steps=1000, alpha=0.5, boundaries=None, start_position=None):
    """
    Generates a 2D Continuous Time Random Walk with optional reflective boundaries.

    Parameters:
    -----------
    steps : int
        Number of steps in the walk.
    alpha : float
        Parameter for the power-law waiting times (alpha > 0).
    boundaries : tuple or None
        (xmin, xmax, ymin, ymax) for reflective boundaries, or None for no boundaries.
    start_position : tuple or None
        Initial position (x_start, y_start) or None for starting at (0,0).

    Returns:
    --------
    tuple of arrays
        (position_x, position_y) containing the trajectory coordinates.
    """
    # Generate random waiting times and jump
    waiting_times = np.random.pareto(alpha, steps) + 1  # Power-law waiting times
    jump_sizes_x = np.random.normal(loc=0, scale=1, size=steps)  # Normal jumps in x
    jump_sizes_y = np.random.normal(loc=0, scale=1, size=steps)  # Normal jumps in y

    position_x = np.zeros(steps)
    position_y = np.zeros(steps)
    if start_position is not None:
        position_x[0], position_y[0] = start_position
    else:
        position_x[0], position_y[0] = 0, 0

    for i in range(1, steps):
        position_x[i] = position_x[i - 1] + jump_sizes_x[i] * waiting_times[i]
        position_y[i] = position_y[i - 1] + jump_sizes_y[i] * waiting_times[i]

        if boundaries:
            xmin, xmax, ymin, ymax = boundaries
            # Check for collisions with x boundaries
            if position_x[i] < xmin:
                position_x[i] = 2 * xmin - position_x[i]  # Reflect back into bounds
                jump_sizes_x[i] = -jump_sizes_x[i]  # Reverse direction
            elif position_x[i] > xmax:
                position_x[i] = 2 * xmax - position_x[i]
                jump_sizes_x[i] = -jump_sizes_x[i]
            # Check for collisions with y boundaries
            if position_y[i] < ymin:
                position_y[i] = 2 * ymin - position_y[i]
                jump_sizes_y[i] = -jump_sizes_y[i]
            elif position_y[i] > ymax:
                position_y[i] = 2 * ymax - position_y[i]
                jump_sizes_y[i] = -jump_sizes_y[i]

    return position_x, position_y



# 4. Levy Walks (2D)
def lw_2d(steps=1000, alpha=1.5, boundaries=None, start_position=None):
    """
    Generates a 2D Levy walk with optional reflective boundaries.

    Parameters:
    -----------
    steps : int
        Number of steps in the walk.
    alpha : float
        Parameter for the power-law distributed step lengths (alpha > 0).
    boundaries : tuple or None
        (xmin, xmax, ymin, ymax) for reflective boundaries, or None for no boundaries.
    start_position : tuple or None
        Initial position (x_start, y_start) or None for starting at (0,0).

    Returns:
    --------
    tuple of arrays
        (position_x, position_y) containing the trajectory coordinates.
    """
    # Generate step lengths and directions
    step_lengths = np.random.pareto(alpha, steps) + 1
    directions = np.random.uniform(0, 2 * np.pi, size=steps)

    # Calculate increments
    increments_x = step_lengths * np.cos(directions)
    increments_y = step_lengths * np.sin(directions)

    # Initialize positions
    position_x = np.zeros(steps)
    position_y = np.zeros(steps)
    if start_position is not None:
        position_x[0], position_y[0] = start_position
    else:
        position_x[0], position_y[0] = 0, 0

    for i in range(1, steps):
        position_x[i] = position_x[i - 1] + increments_x[i]
        position_y[i] = position_y[i - 1] + increments_y[i]
        if boundaries:
            xmin, xmax, ymin, ymax = boundaries
            # Check for collisions with x boundaries
            if position_x[i] < xmin:
                position_x[i] = 2 * xmin - position_x[i]  # Reflect back into bounds
                increments_x[i] = -increments_x[i]  # Reverse direction
            elif position_x[i] > xmax:
                position_x[i] = 2 * xmax - position_x[i]
                increments_x[i] = -increments_x[i]
            # Check for collisions with y boundaries
            if position_y[i] < ymin:
                position_y[i] = 2 * ymin - position_y[i]
                increments_y[i] = -increments_y[i]
            elif position_y[i] > ymax:
                position_y[i] = 2 * ymax - position_y[i]
                increments_y[i] = -increments_y[i]

    return position_x, position_y



# 5. Annealed Transit Time Model (ATTM) (2D)
def attm_2d(steps=1000, kappa=1.5, boundaries=None, start_position=None):
    """
    Generates a 2D Annealed Transit Time Model (ATTM) with optional reflective boundaries.

    Parameters:
    -----------
    steps : int
        Number of steps in the walk.
    kappa : float
        Shape parameter for the gamma distribution (kappa > 0).
    boundaries : tuple or None
        (xmin, xmax, ymin, ymax) for reflective boundaries, or None for no boundaries.
    start_position : tuple or None
        Initial position (x_start, y_start) or None for starting at (0,0).

    Returns:
    --------
    tuple of arrays
        (position_x, position_y) containing the trajectory coordinates.
    """
    diffusivities = np.random.gamma(shape=kappa, scale=1.0, size=steps)
    increments_x = np.random.normal(loc=0, scale=np.sqrt(diffusivities), size=steps)
    increments_y = np.random.normal(loc=0, scale=np.sqrt(diffusivities), size=steps)

    # Initialize positions
    position_x = np.zeros(steps)
    position_y = np.zeros(steps)
    if start_position is not None:
        position_x[0], position_y[0] = start_position
    else:
        position_x[0], position_y[0] = 0, 0

    for i in range(1, steps):
        position_x[i] = position_x[i - 1] + increments_x[i]
        position_y[i] = position_y[i - 1] + increments_y[i]
        if boundaries:
            xmin, xmax, ymin, ymax = boundaries
            # Check for collisions with x boundaries
            if position_x[i] < xmin:
                position_x[i] = 2 * xmin - position_x[i]  # Reflect back into bounds
                increments_x[i] = -increments_x[i]  # Reverse direction
            elif position_x[i] > xmax:
                position_x[i] = 2 * xmax - position_x[i]
                increments_x[i] = -increments_x[i]
            # Check for collisions with y boundaries
            if position_y[i] < ymin:
                position_y[i] = 2 * ymin - position_y[i]
                increments_y[i] = -increments_y[i]
            elif position_y[i] > ymax:
                position_y[i] = 2 * ymax - position_y[i]
                increments_y[i] = -increments_y[i]

    return position_x, position_y


# 6. Scaled Brownian Motion (SBM) (2D)
def sbm_2d(steps=1000, beta=0.5, boundaries=None, start_position=None):
    """
    Generates a 2D Scaled Brownian motion with optional reflective boundaries.

    Parameters:
    -----------
    steps : int
        Number of steps in the walk.
    beta : float
        Scaling exponent for the Brownian motion (0 <= beta <= 1).
    boundaries : tuple or None
        (xmin, xmax, ymin, ymax) for reflective boundaries, or None for no boundaries.
    start_position : tuple or None
        Initial position (x_start, y_start) or None for starting at (0,0).

    Returns:
    --------
    tuple of arrays
        (position_x, position_y) containing the trajectory coordinates.
    """
    time = np.linspace(0, 1, steps)
    scale = time**beta
    increments_x = np.random.normal(loc=0, scale=scale)
    increments_y = np.random.normal(loc=0, scale=scale)
    position_x = np.zeros(steps)
    position_y = np.zeros(steps)
    if start_position is not None:
        position_x[0], position_y[0] = start_position
    else:
        position_x[0], position_y[0] = 0, 0

    for i in range(1, steps):
        position_x[i] = position_x[i - 1] + increments_x[i]
        position_y[i] = position_y[i - 1] + increments_y[i]
        if boundaries:
            xmin, xmax, ymin, ymax = boundaries
            # Check for collisions with x boundaries
            if position_x[i] < xmin:
                position_x[i] = 2 * xmin - position_x[i]  # Reflect back into bounds
                increments_x[i] = -increments_x[i]  # Reverse direction
            elif position_x[i] > xmax:
                position_x[i] = 2 * xmax - position_x[i]
                increments_x[i] = -increments_x[i]
            # Check for collisions with y boundaries
            if position_y[i] < ymin:
                position_y[i] = 2 * ymin - position_y[i]
                increments_y[i] = -increments_y[i]
            elif position_y[i] > ymax:
                position_y[i] = 2 * ymax - position_y[i]
                increments_y[i] = -increments_y[i]

    return position_x, position_y


def create_tif_animation(
    positions, colors, labels, 
    filename="random_walk.tif", frame_size=(1000, 500), 
    dot_radius=5, show_legend=True, boundaries=False, grayscale=False):
    """
    Generates a .tif animation showing multiple 2D random walks with different colors and a legend.
    Allows optional boundary walls and grayscale output.

    Parameters:
    -----------
    positions : list of tuples
        Each tuple contains two lists: x and y positions of the random walks.
    colors : list of str
        Colors of the random walks.
    labels : list of str
        Labels for the legend.
    filename : str
        Output filename for the .tif file.
    frame_size : tuple of int
        Size of each frame (width, height).
    dot_radius : int
        Radius of the dots representing the particles.
    show_legend : bool
        Whether to show a legend in each frame.
    boundaries : bool
        Whether to draw boundary walls around the frames.
    grayscale : bool
        Whether to save the frames in grayscale (True) or in color (False).
    """
    frames = []
    # Skip normalization: use positions as they are
    unnormalized_positions = positions
    # Determine the number of steps from the first walk
    steps = len(positions[0][0])
    for step in tqdm(range(steps), desc="Generating frames"):
        frame = Image.new("RGB", frame_size, "black")
        draw = ImageDraw.Draw(frame)
        for (position_x, position_y), color in zip(unnormalized_positions, colors):
            x, y = position_x[step], position_y[step]
            draw.ellipse([ 
                (x - dot_radius, y - dot_radius),
                (x + dot_radius, y + dot_radius)
            ], fill=color, outline=color)
        # Add boundary walls if specified
        if boundaries:
            draw.rectangle([0, 0, frame_size[0], frame_size[1]], outline="white", width=5)  # Draw a white border around the frame
        # Add legend on every frame if show_legend is True
        if show_legend:
            legend_x_start = 10
            legend_y_start = 10
            legend_spacing = 20
            unique_labels = list(dict.fromkeys(labels))  # Keep only unique labels
            unique_colors = [colors[labels.index(label)] for label in unique_labels]
            for i, label in enumerate(unique_labels):
                legend_y = legend_y_start + i * legend_spacing
                draw.rectangle([
                    (legend_x_start, legend_y),
                    (legend_x_start + 10, legend_y + 10)
                ], fill=unique_colors[i], outline=unique_colors[i])
                draw.text((legend_x_start + 15, legend_y), label, fill="white")
        # Convert the frame to grayscale if specified
        if grayscale:
            frame = frame.convert("L")
        frames.append(frame)
    # Save as .tif animation
    frames[0].save(
        filename, save_all=True, append_images=frames[1:], duration=50, loop=0)
    print(f"Animation saved as {filename} \n")


def tif_to_mp4(tif_input, output_video, fps=30):
    """
    Converts a multi-frame .tif file to an MP4 video with a progress bar.

    Parameters:
    ----------
    tif_input : str
        Path to the .tif file containing the frames.
    output_video : str
        Output path for the MP4 video.
    fps : int, optional
        Framerate (frames per second) of the video. Default is 30.
    """
    frames = tiff.imread(tif_input)
    print(f"Number of frames detected: {len(frames)}")
    if len(frames[0].shape) == 3:
        height, width, channels = frames[0].shape
    elif len(frames[0].shape) == 2:
        height, width = frames[0].shape
        channels = 1
    else:
        raise ValueError("Unexpected frame structure in the .tif file")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Add each frame to the video with a progress bar
    for i, frame in tqdm(enumerate(frames), total=len(frames), desc="Creating video", ncols=100):
        frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame_normalized = frame_normalized.astype('uint8')
        if channels == 1:  # If the image is grayscale
            frame_normalized = cv2.cvtColor(frame_normalized, cv2.COLOR_GRAY2BGR)
        video_writer.write(frame_normalized)
    video_writer.release()
    print(f"MP4 video generated: {output_video}")
    
    
def generate_and_save_walks(walk_type, steps=1000, num_walks=1000, filename="random_walks.csv", start_positions=None, **kwargs):
    """
    Generates multiple random walks and saves their trajectories to a CSV file.

    Parameters:
    -----------
    walk_type : function
        Function to generate the walk (e.g., Brownian motion, Levy walk).
    steps : int
        Number of steps in each walk.
    num_walks : int
        Number of walks to generate.
    filename : str
        Output CSV filename to save the walk data.
    start_positions : list of tuples or None
        Initial positions for each walk. If None, random positions are generated.
    **kwargs : additional parameters
        Additional parameters for the walk generation function.

    The function generates random walks using the specified walk_type function,
    records the positions at each step, and saves the data to a CSV file.
    """
    data = []
    if start_positions is None:
        start_positions = [(np.random.uniform(0, 1000), np.random.uniform(0, 500)) for _ in range(num_walks)]
    for n in tqdm(range(num_walks), desc="Generating random walks"):
        x, y = walk_type(steps=steps, start_position=start_positions[n], **kwargs)
        for frame in range(steps):
            data.append([x[frame], y[frame], n, frame * 1.0, frame])
    # Convert data to DataFrame and save to CSV
    df = pd.DataFrame(data, columns=["x", "y", "n", "t", "frame"])
    df.to_csv(filename, index=False)
    print(f"CSV file saved: {filename}")



"""
#Exemple d'utilisation avec des positions de départ homogènes
if __name__ == "__main__":
    # Parameters
    steps = 500
    num_bm = 1000  # Number of Brownian Motion particles
    boundaries = (5, 295, 5, 195)  # Image boundaries
    size = (boundaries[1] - boundaries[0] + 5, boundaries[3] - boundaries[2] + 5)
    start_zone = boundaries  # Uniform start zone

    # Generate random uniform start positions within the start zone
    x_starts = np.random.uniform(start_zone[0], start_zone[1], num_bm)
    y_starts = np.random.uniform(start_zone[2], start_zone[3], num_bm)
    start_positions = [(x, y) for x, y in zip(x_starts, y_starts)]

    # Generate and save random walks with boundaries
    csv_filename_with_boundaries = "with_boundaries_brownian_motion_walks.csv"
    generate_and_save_walks(bm_2d, steps=steps, num_walks=num_bm, filename=csv_filename_with_boundaries, boundaries=boundaries, start_positions=start_positions, dt=1.0)

    # Generate and save random walks without boundaries
    csv_filename_without_boundaries = "without_boundaries_brownian_motion_walks.csv"
    generate_and_save_walks(bm_2d, steps=steps, num_walks=num_bm, filename=csv_filename_without_boundaries, boundaries=None, start_positions=start_positions, dt=1.0)

    # Read CSV files and generate positions for animations
    df_with_boundaries = pd.read_csv(csv_filename_with_boundaries)
    df_without_boundaries = pd.read_csv(csv_filename_without_boundaries)

    # Create TIF animations
    positions = []
    colors = []
    labels = []

    # Add positions for walks with boundaries
    for n in range(num_bm):
        positions.append((df_with_boundaries[df_with_boundaries['n'] == n]['x'].values, df_with_boundaries[df_with_boundaries['n'] == n]['y'].values))
        colors.append("white")
        labels.append("Brownian Motion")

    # Create .tif file with boundaries
    output_filename_with_boundaries = "with_boundaries_random_walk.tif"
    create_tif_animation(positions, colors, labels, frame_size=size, filename=output_filename_with_boundaries, show_legend=False, boundaries=boundaries, grayscale=True)

    # Convert to MP4 video with boundaries
    tif_to_mp4(output_filename_with_boundaries, "with_boundaries_random_walk.mp4", fps=20)

    # Reset lists for walks without boundaries
    positions = []
    colors = []
    labels = []

    # Add positions for walks without boundaries
    for n in range(num_bm):
        positions.append((df_without_boundaries[df_without_boundaries['n'] == n]['x'].values, df_without_boundaries[df_without_boundaries['n'] == n]['y'].values))
        colors.append("white")
        labels.append("Brownian Motion")

    # Create .tif file without boundaries
    output_filename_without_boundaries = "without_boundaries_random_walk.tif"
    create_tif_animation(positions, colors, labels, frame_size=size, filename=output_filename_without_boundaries, show_legend=False, boundaries=None, grayscale=True)

    # Convert to MP4 video without boundaries
    tif_to_mp4(output_filename_without_boundaries, "without_boundaries_random_walk.mp4", fps=20)"""