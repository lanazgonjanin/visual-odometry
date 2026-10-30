import cv2
import numpy as np

from src.dataset import load_image, load_calibration, get_intrinsic_matrix
from src.features.orb import detect_orb_features
from src.features.matching import match_features
from src.geometry.essential import extract_matched_points, estimate_essential_matrix


# Load the first two KITTI frames
image1 = load_image("data/sequences/00/image_0/000000.png")
image2 = load_image("data/sequences/00/image_0/000001.png")

# Detect and match ORB features
keypoints1, descriptors1 = detect_orb_features(image1)
keypoints2, descriptors2 = detect_orb_features(image2)
matches = match_features(descriptors1, descriptors2)

# Extract coordinates and estimate the essential matrix
points1, points2 = extract_matched_points(keypoints1, keypoints2, matches)

calibration = load_calibration("data/calib/00/calib.txt")
K = get_intrinsic_matrix(calibration)

E, mask = estimate_essential_matrix(points1, points2, K)

# Flatten the mask from shape (N, 1) to (N,)
inlier_mask = mask.ravel()

# Separate matches based on the RANSAC mask
inlier_matches = [
    match for match, is_inlier in zip(matches, inlier_mask)
    if is_inlier
]

outlier_matches = [
    match for match, is_inlier in zip(matches, inlier_mask)
    if not is_inlier
]

# Draw up to 50 inlier matches in green
inlier_visualization = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    inlier_matches[:50],
    None,
    matchColor=(0, 255, 0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

# Draw up to 50 outlier matches in red
outlier_visualization = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    outlier_matches[:50],
    None,
    matchColor=(0, 0, 255),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

# Combine the visualizations vertically
combined = np.vstack((inlier_visualization, outlier_visualization))

# Add labels to distinguish the panels
cv2.putText(
    combined, "RANSAC Inliers", (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
)

cv2.putText(
    combined, "RANSAC Outliers", (20, image1.shape[0] + 35),
    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
)

# Save the result
cv2.imwrite("results/ransac_matches.png", combined)

print(f"RANSAC inliers: {len(inlier_matches)}")
print(f"RANSAC outliers: {len(outlier_matches)}")
print("Saved visualization to results/ransac_matches.png")
