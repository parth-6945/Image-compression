import cv2
import os
import time
import numpy as np
from collections import Counter

def calculate_entropy(image):
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

def jpeg2000_compression(image_path, compressed_path):
    start_time = time.time()

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Compress the image using OpenCV (JPEG 2000)
    result = cv2.imwrite(compressed_path, image, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 1000])

    if not result:
        raise ValueError("Compression failed")

    end_time = time.time()

    # Get sizes and calculate compression ratio
    original_size = os.path.getsize(image_path)
    compressed_size = os.path.getsize(compressed_path)
    compression_ratio = original_size / compressed_size
    time_taken_compress = end_time - start_time

    # Load the compressed image to calculate entropy and redundancy
    compressed_image = cv2.imread(compressed_path, cv2.IMREAD_UNCHANGED)
    if compressed_image is None:
        raise FileNotFoundError(f"Failed to load compressed image from {compressed_path}")

    # Calculate entropy and redundancy for the compressed image
    entropy, redundancy = calculate_entropy(compressed_image)

    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": compression_ratio,
        "time_taken": time_taken_compress,
        "entropy": entropy,
        "redundancy": redundancy,
    }

def jpeg2000_decompression(compressed_path, output_path):
    # Decompress the image using OpenCV
    image = cv2.imread(compressed_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(f"Failed to load compressed image from {compressed_path}")

    # Save the decompressed image
    cv2.imwrite(output_path, image)
    return image

if __name__ == "__main__":
    input_image_path = "Images/480-360-sample.bmp"  # Replace with your image path
    output_compressed_path = "compressed_image.jp2"  # JPEG 2000 file extension
    decompressed_image_path = "decompressed_image.png"  # Decompressed output path

    # Compression
    result = jpeg2000_compression(input_image_path, output_compressed_path)
    print("Compression Result:", result)

    # Decompression
    decompressed_image = jpeg2000_decompression(output_compressed_path, decompressed_image_path)
    print("Decompression complete.")
