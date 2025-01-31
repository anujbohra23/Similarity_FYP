import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def apply_gradient_filter(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    gradient_magnitude = cv2.normalize(
        gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX
    )
    gradient_magnitude = np.uint8(gradient_magnitude)

    return gradient_magnitude


def enhance_contrast(image):
    min_val, max_val = np.min(image), np.max(image)
    contrast_stretched = (image - min_val) * (255 / (max_val - min_val))
    contrast_stretched = np.uint8(contrast_stretched)

    return contrast_stretched


def mask_and_threshold(image):
    _, binary_image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
    inverted_binary = cv2.bitwise_not(binary_image)

    return inverted_binary


def process_image(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    gradient_image = apply_gradient_filter(image)
    enhanced_image = enhance_contrast(gradient_image)
    masked_image = mask_and_threshold(enhanced_image)

    # Negate the masked image
    negated_image = cv2.bitwise_not(masked_image)

    # Generate and save the figure
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 5, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(cv2.cvtColor(gradient_image, cv2.COLOR_GRAY2RGB))
    plt.title("Gradient Image")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB))
    plt.title("Enhanced Gradient Image")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_GRAY2RGB))
    plt.title("Masked & Thresholded Image")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(cv2.cvtColor(negated_image, cv2.COLOR_GRAY2RGB))
    plt.title("Negated Image")
    plt.axis("off")

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
        r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\anujwarp\addNegatedAndThresholdedImages"
    )

    main(input_directory, output_directory)
