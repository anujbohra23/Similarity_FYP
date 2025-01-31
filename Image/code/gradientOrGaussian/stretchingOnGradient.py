import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def apply_sobel_filter(image):
    # Sobel Filte
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_8u = np.uint8(np.absolute(sobel_combined))

    return sobel_8u  # 8 bit image for stretching


def contrast_stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)

    stretched = (image - min_val) * (255.0 / (max_val - min_val))
    stretched = np.uint8(stretched)

    return stretched


def blur_and_highlight_edges(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # Applying filter
    gradient_image = apply_sobel_filter(image)
    stretched_image = contrast_stretching(gradient_image)
    blurred_image = cv2.GaussianBlur(stretched_image, (51, 51), 0)
    edges = cv2.Canny(blurred_image, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    highlighted_image = np.where(edges_colored == 255, 255, blurred_image[:, :, None])

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
    plt.title("Blurred & Highlighted Edges")
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
            blur_and_highlight_edges(image_path, output_directory)


if __name__ == "__main__":
    input_directory = (
        r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\Dataset\IIT Patna Dataset\AngledImages"
    )
    output_directory = (
        r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\anujwarp\contrastStretchingImages"
    )

    main(input_directory, output_directory)
