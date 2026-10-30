import cv2
import numpy as np

# Each keypoint has a 256-bit binary descriptor describing the visual appearance around that point
# To check whether two keypoints match, we compare their descriptors using Hamming distance
# Hamming distance compares the two 256-bit binary strings and returns the number of bits whose binary values don't match
# Smaller Hamming distance --> better match

def match_features(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
) -> list[cv2.DMatch]:

    # Compute Hamming dist between every descriptor in image 1 and every descriptor in image 2
    # Find best match of each descriptor by finding descriptor in other image with smallest Hamming dist, ensuring descriptor A's best match is B and B's best match is A
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

    matches = sorted(matches, key=lambda match: match.distance)  # order matches from smallest to largest

    return matches

