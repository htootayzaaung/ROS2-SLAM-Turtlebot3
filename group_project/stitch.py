import cv2

def stitch_images():
    # Load the images
    image1 = cv2.imread('window0.png')
    image2 = cv2.imread('window1.png')

    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize the stitcher
    stitcher = cv2.Stitcher_create()

    # Stitch the images
    status, stitched_image = stitcher.stitch([image1, image2])

    if status==cv2.STITCHER_OK:
        cv2.imwrite('Panorama.jpg', stitched_image)
    else:
        print("Stitching failed!")
