import cv2
import numpy as np

# Load the images
earth_img = cv2.imread('framed_earth.jpg')
moon_img = cv2.imread('framed_moon.jpg')

def crop_via_circle_detection(image, min_radius, max_radius):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a Gaussian blur to reduce noise and improve circle detection
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=100, param2=30, minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Take the first detected circle (assuming it's the primary object)
        x, y, radius = circles[0]
        # Crop the image to the bounding box of the circle
        x_start = max(0, x - radius)
        y_start = max(0, y - radius)
        x_end = min(image.shape[1], x + radius)
        y_end = min(image.shape[0], y + radius)
        cropped = image[y_start:y_end, x_start:x_end]
        return cropped
    else:
        return image

# Detect and crop the Earth and Moon images with reasonable radius ranges
cropped_earth = crop_via_circle_detection(earth_img, min_radius=50, max_radius=300)
cropped_moon = crop_via_circle_detection(moon_img, min_radius=20, max_radius=150)

# Save or display the cropped images
cv2.imwrite('cropped_earth_v2.jpg', cropped_earth)
cv2.imwrite('cropped_moon_v2.jpg', cropped_moon)

