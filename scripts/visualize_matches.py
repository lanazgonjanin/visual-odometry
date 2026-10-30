import cv2

from src.dataset import load_image
from src.features.orb import detect_orb_features
from src.features.matching import match_features

# Load two consecutive frames
image1 = load_image("data/sequences/00/image_0/000000.png")
image2 = load_image("data/sequences/00/image_0/000001.png")

# Detect ORB features in both frames
keypoints1, descriptors1 = detect_orb_features(image1)
keypoints2, descriptors2 = detect_orb_features(image2)

# Match descriptors between frames
matches = match_features(descriptors1, descriptors2)

# Visualize the 50 best matches, drawing a line connecting the matching keypoints
visualization = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    matches[:50],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

cv2.imwrite("results/orb_matches.png", visualization)

print(f"Total matches: {len(matches)}")
print(f"Visualised {min(50, len(matches))} matches.")
