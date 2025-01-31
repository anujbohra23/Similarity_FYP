import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def blur_text_area(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # Create ROI coordinates (modify these coordinates as per your requirement)
    topLeft = (60, 140)
    bottomRight = (340, 250)
    x, y = topLeft[0], topLeft[1]
    w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]

    # Grab ROI with Numpy slicing and blur
    ROI = image[y : y + h, x : x + w]
    blur = cv2.GaussianBlur(ROI, (51, 51), 0)

    # Insert ROI back into image
    image[y : y + h, x : x + w] = blur

    # Detect edges using Canny edge detection
    edges = cv2.Canny(image, 50, 150)

    # Convert edges to a 3-channel image
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Highlight edges in the original image
    highlighted_image = np.where(edges_colored == 255, edges_colored, image)

    # Generate and save the figure
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Blurred Text Area")
    plt.axis("off")

    # Image with highlighted edges
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
    plt.title("Edges Highlighted")
    plt.axis("off")

    # Save the figure
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file = os.path.join(output_path, f"{base_name}_blurred_edges.png")
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

    print(f"Image saved to {output_file}")


def main(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith((".jpg", ".png", ".jpeg", ".bmp", ".tiff")):
            image_path = os.path.join(input_directory, filename)
            blur_text_area(image_path, output_directory)


if __name__ == "__main__":
    input_directory = (
        r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\Dataset\IIT Patna Dataset\AngledImages"
    )
    output_directory = (
        r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\anujwarp\blurred_highlighted_images"
    )

    main(input_directory, output_directory)
