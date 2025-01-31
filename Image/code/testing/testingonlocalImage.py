from skimage.io import imread, imshow
from skimage import transform
import matplotlib.pyplot as plt
import numpy as np

img = imread(r"C:\Users\Anuj Bohra\Downloads\Ashvagandha.jpg")
plt.figure()
imshow(img)
plt.title("Original Image")
plt.show()

# Define source and destinationcoordinates
source = np.array([391, 100, 14, 271, 347, 624, 747, 298]).reshape((4, 2))
destination = np.array([100, 100, 100, 650, 650, 650, 650, 100]).reshape((4, 2))

# Estimate the transformation using projective
tform = transform.estimate_transform("projective", source, destination)
tf_img = transform.warp(img, tform.inverse)

plt.figure()
plt.imshow(tf_img)
plt.title("Projective Transformation")
plt.show()

# Print the title object
sdsds = plt.gca().get_title()
print(sdsds)
