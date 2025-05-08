import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load the image
image = cv2.imread('img.jpeg')

# Apply Gaussian Blur
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Apply sharpening filter
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpened_image = cv2.filter2D(image, -1, sharpen_kernel)

# Edge detection on the blurred image
edges = cv2.Canny(blurred_image, 50, 150)

# Display results
cv2_imshow(image)
cv2_imshow(blurred_image)
cv2_imshow(sharpened_image)
cv2_imshow(edges)


