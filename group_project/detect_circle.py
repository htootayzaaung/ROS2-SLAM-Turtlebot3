import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError


def Detect_Circle(image, sensitivity):
    bridge = CvBridge()
    try:
        image = bridge.imgmsg_to_cv2(image, 'bgr8')  # Convert to BGR format
    except CvBridgeError as e:
        print(f"Error converting image: {e}")
        return None

    # Set color ranges in HSV for green and red detection
    hsv_red_lower = np.array([0 - sensitivity, 100, 100])
    hsv_red_upper = np.array([0 + sensitivity, 255, 255])
    hsv_green_lower = np.array([60 - sensitivity, 100, 100])
    hsv_green_upper = np.array([60 + sensitivity, 255, 255])

    # Convert image to HSV and grayscale formats
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create masks for green and red colors using inRange
    green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
    red_mask = cv2.inRange(hsv_image, hsv_red_lower, hsv_red_upper)

    # Apply masks to grayscale image to isolate color regions
    green_gray = cv2.bitwise_and(gray_image, gray_image, mask=green_mask)
    red_gray = cv2.bitwise_and(gray_image, gray_image, mask=red_mask)

    # Apply Gaussian blur for noise reduction
    green_blurred = cv2.GaussianBlur(green_gray, (5, 5), 0)
    red_blurred = cv2.GaussianBlur(red_gray, (5, 5), 0)

    # Detect circles using Hough Circle Transform for green and red
    green_circles = cv2.HoughCircles(green_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                    param1=200, param2=30, minRadius=0, maxRadius=0)
    red_circles = cv2.HoughCircles(red_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                   param1=200, param2=30, minRadius=0, maxRadius=0)

    green_circles_detected = green_circles is not None
    red_circles_detected = red_circles is not None

    return green_circles_detected, red_circles_detected
