import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def apply_scharr_filter(image):
    # Compute gradients along the X and Y axis using Scharr filter
    grad_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)

    # Compute the gradient magnitude
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalize the gradient magnitude to the range 0-255
    gradient_magnitude = cv2.normalize(
        gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX
    )
    gradient_magnitude = np.uint8(gradient_magnitude)

    return gradient_magnitude


def blur_and_highlight_edges(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # Apply the Scharr filter
    gradient_image = apply_scharr_filter(image)

    # Apply Gaussian blur to the gradient image
    blurred_image = cv2.GaussianBlur(gradient_image, (51, 51), 0)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(blurred_image, 50, 150)

    # Convert edges to a 3-channel image
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Highlight edges in the original image
    highlighted_image = np.where(edges_colored == 255, 255, blurred_image[:, :, None])

    # Generate and save the figure
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    plt.title("Original Image")
    plt.axis("off")

    # Blurred and highlighted edges
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
    plt.title("Blurred & Highlighted Edges")
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
            blur_and_highlight_edges(image_path, output_directory)


if __name__ == "__main__":
    input_directory = (
        r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\Dataset\IIT Patna Dataset\AngledImages"
    )
    output_directory = (
        r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\anujwarp\gradientToGaussianScharrImages"
    )

    main(input_directory, output_directory)
