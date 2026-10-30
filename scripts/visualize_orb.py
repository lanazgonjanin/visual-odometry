import cv2

from src.dataset import load_image
from src.features.orb import detect_orb_features


image = load_image("data/sequences/00/image_0/000000.png")

keypoints, descriptors = detect_orb_features(image)

# Draw circles at detected keypoint locations
visualization = cv2.drawKeypoints(
    image,
    keypoints,
    None,  # create new output image instead of modifying existing one
    color=(0, 255, 0),  # set colour of keypoint markings to gree
    flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
)

cv2.imwrite("results/orb_keypoints.png", visualization)

print(f"Visualised {len(keypoints)} keypoints.")

