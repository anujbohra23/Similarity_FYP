import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def apply_gradient_filter(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the gradient magnitude
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalize the gradient magnitude
    gradient_magnitude = cv2.normalize(
        gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX
    )
    gradient_magnitude = np.uint8(gradient_magnitude)

    return gradient_magnitude


def threshold_to_binary(image):
    # Apply binary thresholding
    _, binary_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    return binary_image


def process_image(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # Apply the gradient filter
    gradient_image = apply_gradient_filter(image)

    # Apply binary thresholding to the gradient image
    binary_image = threshold_to_binary(gradient_image)

    # Generate and save the figure
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    plt.title("Original Image")
    plt.axis("off")

    # Gradient image
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(gradient_image, cv2.COLOR_GRAY2RGB))
    plt.title("Gradient Image")
    plt.axis("off")

    # Binary image
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB))
    plt.title("Binary Image")
    plt.axis("off")

    # Save the figure
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file = os.path.join(output_path, f"{base_name}_processed.png")
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

    print(f"Image saved to {output_file}")


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
        r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\anujwarp\gradientToBinaryImages"
    )

    main(input_directory, output_directory)
