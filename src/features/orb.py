import cv2
import numpy as np

# ORB is an algorithm that detects distinctive keypoints to be tracked across a sequence of images
# It uses descriptors, which are numerical representations of the visual appearance around keypoints, used to identify and match the same features across different images

def detect_orb_features(
    image: np.ndarray,
    nfeatures: int = 2000,  # max number of keypoints ORB retains
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    orb = cv2.ORB_create(nfeatures=nfeatures)  # create ORB feature detector

    keypoints, descriptors = orb.detectAndCompute(image, None)  # detect keypoints and compute their descriptor (second argument "None" indicates no image region restriction)

    # keypoints: a collection of OpenCV KeyPoint objects
    # descriptors: a NumPy array containing one binary descriptor per keypoint (stored as bytes)

    return keypoints, descriptors

