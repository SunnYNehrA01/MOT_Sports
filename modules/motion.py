import cv2
import numpy as np

class CameraMotionCompensation:
    def __init__(self):
        # ORB is fast and efficient for real-time tracking
        self.detector = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def compute_motion(self, prev_frame, curr_frame):
        """
        Estimates the affine transform between two frames.
        Returns: 2x3 Affine Matrix (M)
        """
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Detect and compute
        kp1, des1 = self.detector.detectAndCompute(prev_gray, None)
        kp2, des2 = self.detector.detectAndCompute(curr_gray, None)

        if des1 is None or des2 is None:
            return np.eye(2, 3, dtype=np.float32)

        # Match features
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract location of good matches
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        if len(pts1) > 10:
            # RANSAC finds the most likely camera shift, ignoring moving players
            M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)
            if M is not None:
                return M
        
        return np.eye(2, 3, dtype=np.float32)