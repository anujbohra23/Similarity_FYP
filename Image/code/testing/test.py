import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\IMG_3524.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# defining points in local and transformed imgges
src_points = np.array([[50, 50], [200, 50], [200, 200], [50, 200]], dtype=np.float32)
dst_points = np.array([[10, 100], [200, 50], [220, 250], [100, 300]], dtype=np.float32)

# Calculating the homography matrix using findHomography
H, status = cv2.findHomography(src_points, dst_points)

# Get the dimensions of the image
height, width, channels = image.shape

# Applying Homography to get warpPerspective
warped_image = cv2.warpPerspective(image, H, (width, height))

# Display the original and warped images
plt.subplot(121), plt.imshow(image), plt.title("Original Image")
plt.subplot(122), plt.imshow(warped_image), plt.title("Warped Image")
plt.show()
