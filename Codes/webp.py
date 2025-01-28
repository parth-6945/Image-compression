import cv2
from PIL import Image
import numpy as np
import os
import time
from collections import Counter


def calculate_entropy(image):
    """
    Calculates the entropy and redundancy of an image.
    
    :param image: The input image as a numpy array.
    :return: Entropy and redundancy values.
    """
    # Flatten the image and count the occurrences of each pixel value
    flat_image = image.flatten()
    value_counts = Counter(flat_image)
    total_pixels = len(flat_image)

    # Calculate probabilities of unique pixel values
    probabilities = [count / total_pixels for count in value_counts.values()]

    # Calculate entropy
    entropy = -sum(p * np.log2(p) for p in probabilities)

    # Calculate maximum entropy for an 8-bit image (256 possible values)
    max_entropy = np.log2(256)

    # Calculate redundancy
    redundancy = max_entropy - entropy

    return entropy, redundancy


def webp_compression(image_path, output_path):
    """
    Compresses an image using WebP in lossless mode (quality=100).
    
    :param image_path: Path to the input image.
    :param output_path: Path to save the WebP-compressed image.
    :return: Compression statistics.
    """
    start_time = time.time()

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Save the image as WebP with quality=100 (lossless)
    is_success = cv2.imwrite(output_path, image, [cv2.IMWRITE_WEBP_QUALITY, 100])
    if not is_success:
        raise IOError(f"Failed to save image as WebP at {output_path}")

    end_time = time.time()

    # Get sizes and calculate compression ratio
    original_size = os.path.getsize(image_path)
    compressed_size = os.path.getsize(output_path)
    compression_ratio = original_size / compressed_size
    time_taken = end_time - start_time

    # Load the compressed image to calculate entropy and redundancy
    compressed_image = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
    if compressed_image is None:
        raise FileNotFoundError(f"Failed to load compressed image from {output_path}")

    # Calculate entropy and redundancy for the compressed image
    entropy, redundancy = calculate_entropy(compressed_image)

    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": compression_ratio,
        "time_taken": time_taken,
        "entropy": entropy,
        "redundancy": redundancy,
    }


def webp_decompression(webp_path, output_image_path):
    """
    Decompresses a WebP image back to its original format.
    
    :param webp_path: Path to the WebP-compressed image.
    :param output_image_path: Path to save the decompressed image.
    :return: Decompressed image as a numpy array.
    """
    # Open the WebP file with Pillow
    image = Image.open(webp_path)
    image = image.convert("RGB")  # Convert to RGB to save as a standard image

    # Save the decompressed image
    image.save(output_image_path)

    # Convert the image to a numpy array for comparison
    return cv2.imread(output_image_path)


if __name__ == "__main__":
    input_image_path = "Images/480-360-sample.bmp"  # Replace with your image path
    compressed_webp_path = "compressed_image.webp"
    decompressed_image_path = "decompressed_image.jpg"

    # WebP Compression
    compression_result = webp_compression(input_image_path, compressed_webp_path)
    print("WebP Compression Results:", compression_result)

    # WebP Decompression
    decompressed_image = webp_decompression(compressed_webp_path, decompressed_image_path)

    # Verify lossless decompression
    original_image = cv2.imread(input_image_path)

    if np.array_equal(original_image, decompressed_image):
        print("Decompression verified: Original and decompressed images match!")
    else:
        print("Decompression failed: Original and decompressed images do not match.")
