import cv2
import numpy as np
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


def contrast_stretching(image):
    # Apply contrast stretching
    min_val = np.min(image)
    max_val = np.max(image)
    stretched_image = (image - min_val) * (255 / (max_val - min_val))
    stretched_image = np.uint8(stretched_image)

    return stretched_image


def threshold_image(gradient_image):
    _, binary_image = cv2.threshold(
        gradient_image, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary_image


def find_and_mask_products(binary_image, original_image, output_directory, base_name):
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for i, contour in enumerate(contours):
        mask = np.ones_like(original_image) * 255
        cv2.drawContours(mask, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)

        output_file = os.path.join(output_directory, f"{base_name}_product_{i + 1}.png")
        cv2.imwrite(output_file, mask)
        print(f"Product {i + 1} mask saved to {output_file}")


def process_image(image_path, output_directory):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    gradient_image = apply_gradient_filter(image)
    stretched_image = contrast_stretching(gradient_image)
    binary_image = threshold_image(stretched_image)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    find_and_mask_products(binary_image, image, output_directory, base_name)


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
    output_directory = r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\anujwarp\productMasks"

    main(input_directory, output_directory)
