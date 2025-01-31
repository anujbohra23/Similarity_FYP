import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


def apply_gaussian_blur(image, kernel_size=5, sigma=1):
    """
    kernel_size: Size of the Gaussian kernel. Must be odd.
    sigma: Standard deviation of the Gaussian bistribution.

    output - Blurred image.
    """

    blurred_image = cv.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred_image


def apply_sobel_operator(image):
    grad_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)

    # Compute the gradient magnitude
    grad_magnitude = cv.magnitude(grad_x, grad_y)

    # Convert to 8-bit image
    grad_magnitude = np.uint8(grad_magnitude)

    return grad_magnitude


def process_image(image_path, output_path):
    # Load the image
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Apply Gaussian blur
    blurred_image = apply_gaussian_blur(image)

    # Apply Sobel gradient
    gradient_image = apply_sobel_operator(blurred_image)

    # Display the images
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(image, cmap="gray"), plt.title(
        "Original Image"
    ), plt.axis("off")
    plt.subplot(132), plt.imshow(blurred_image, cmap="gray"), plt.title(
        "Gaussian Blurred"
    ), plt.axis("off")
    plt.subplot(133), plt.imshow(gradient_image, cmap="gray"), plt.title(
        "Sobel Gradient"
    ), plt.axis("off")
    # Save the figure
    base_name = os.path.basename(image_path)
    output_file = os.path.join(
        output_directory, f"{os.path.splitext(base_name)[0]}_gradient.png"
    )
    plt.savefig(output_file, bbox_inches="tight")


def main(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith((".jpg", ".png", ".jpeg", ".bmp", ".tiff")):
            image_path = os.path.join(input_directory, filename)
            process_image(image_path, output_directory)


if __name__ == "__main__":
    input_directory = (
        r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\Dataset\IIT Patna Dataset\AngledImages"
    )
    output_directory = (
        r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\anujwarp\gaussianToGradientImages"
    )
    main(input_directory, output_directory)
