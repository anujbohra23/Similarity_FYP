import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def process_image(image_path, output_directory):
    """
    Process a single image, apply Gaussian blur, and edge detection, then save the output.
    """
    src = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if src is None:
        print(f"Error opening image {image_path}!")
        return

    # Apply Gaussian Blur
    blurred = cv.GaussianBlur(src, (5, 5), 0)

    # Edge detection using Canny
    edges = cv.Canny(blurred, 50, 150)

    # Using matplotlib to display images
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(src, cmap="gray"), plt.title("Source Image"), plt.axis(
        "off"
    )
    plt.subplot(132), plt.imshow(blurred, cmap="gray"), plt.title(
        "Gaussian Blurred"
    ), plt.axis("off")
    plt.subplot(133), plt.imshow(edges, cmap="gray"), plt.title(
        "Edge Detection"
    ), plt.axis("off")

    # Save the figure
    base_name = os.path.basename(image_path)
    output_file = os.path.join(
        output_directory, f"{os.path.splitext(base_name)[0]}_edges.png"
    )
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()


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
    output_directory = r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\anujwarp\EdgeImages"
    main(input_directory, output_directory)
