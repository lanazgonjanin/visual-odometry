from pathlib import Path

import numpy as np
import cv2

# Each sequence of images is recorded using 4 cameras (P0: left greyscale, P1: right greyscale, P2: left colour, P3: right colour)
# The calibration of each camera is constant throughout the sequence and is described by projection matrix Px (where x denotes the camera)
# Calibration tells us how the camera projects a 3D point to a 2D pixel

# Each calibration file contains "projection matrix" of the four cameras in the form of a row of numbers. Convert the rows of numbers into a NumPy matrix
def load_calibration(calibration_path: str | Path) -> dict[str, np.ndarray]:
    calibration = {}

    with open(calibration_path, "r") as file:
        for line in file:
            parts = line.strip().split()

            if not parts:
                continue

            name = parts[0].rstrip(":")
            values = np.array(parts[1:], dtype=np.float64)

            if values.size != 12:
                raise ValueError(f"Expected 12 values for {name}, got {values.size}")

            calibration[name] = values.reshape(3, 4)

    return calibration


# For this project, we are only looking at left greyscale camera, so only need P0
# Intrinsic matrix is the internal parameters (focal length and principal point) of the projection matrix
def get_intrinsic_matrix(calibration: dict[str, np.ndarray]) -> np.ndarray:
    return calibration["P0"][:, :3]


# For every sequence of images, ground-truth file contains the camera's position and orientation relative to the starting coordinate frame
# Read each ground-truth camera pose and convert into 4x4 transformation matrix representing the camera's position and orientation
def load_poses(poses_path: str | Path) -> np.ndarray:

    poses = np.loadtxt(poses_path, dtype=np.float64)

    poses = poses.reshape(-1, 3, 4)  # reshape each row of 12 values into 3x4 matrix

    transforms = np.tile(np.eye(4), (len(poses), 1, 1))  # create one 4x4 identity matrix for each pose

    transforms[:, :3, :] = poses  # insert each 3x4 matrix into its 4x4 identity (4x4 matrix lets us perform matrix manipulation easier)

    return transforms  # return N 4x4 matrices, where N is the number of poses in the sequence


# Load the timestamp of each pose/frame (in seconds)
def load_timestamps(times_path: str | Path) -> np.ndarray:
    timestamps = np.loadtxt(times_path, dtype=np.float64)
    return np.atleast_1d(timestamps)


# Return image paths sorted by frame number
def get_image_paths(image_dir: str | Path) -> list[Path]:
    image_dir = Path(image_dir)

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted(
        path
        for path in image_dir.glob("*.png")
        if not path.name.startswith("._")
    )

    if not image_paths:
        raise ValueError(f"No PNG images found in {image_dir}")

    return image_paths


# Load greyscale image as a NumPy array
def load_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return image