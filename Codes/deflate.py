import cv2
import zlib
import numpy as np
import time
from math import log2


def calculate_entropy(image):
    """
    Calculate the entropy of an image.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        float: Entropy of the image.
    """
    # Flatten the image to a 1D array
    flat_image = image.flatten()

    # Count the frequency of each pixel value
    unique, counts = np.unique(flat_image, return_counts=True)
    probabilities = counts / counts.sum()

    # Calculate entropy
    entropy = -sum(p * log2(p) for p in probabilities if p > 0)
    return entropy


def compress_image_with_deflate(image_path):
    """
    Compresses an image using the Deflate algorithm and calculates performance metrics.

    Args:
        image_path (str): Path to the input image.

    Returns:
        dict: Metrics including original size, compressed size, compression ratio, 
              time taken for compression, entropy, and redundancy.
    """
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")

    # Calculate entropy
    entropy = calculate_entropy(image)

    # Flatten the image into a 1D byte array
    image_data = image.tobytes()

    # Record original size in bytes
    original_size = len(image_data)

    # Start compression timing
    start_time = time.time()

    # Compress the byte data using Deflate (zlib)
    compressed_data = zlib.compress(image_data, level=9)

    # End compression timing
    time_taken_compress = time.time() - start_time

    # Record compressed size in bytes
    compressed_size = len(compressed_data)

    # Calculate compression ratio
    compression_ratio = original_size / compressed_size if compressed_size > 0 else float("inf")

    # Calculate redundancy
    redundancy = 1 - (entropy / (log2(256) * compression_ratio)) if compression_ratio > 0 else 0

    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": compression_ratio,
        "time_taken_compress": time_taken_compress,
        "entropy": entropy,
        "redundancy": redundancy
    }


if __name__ == "__main__":
    # Path to your image
    image_path = "Wallpaper_1.jpg"

    # Compress the image and retrieve metrics
    metrics = compress_image_with_deflate(image_path)

    # Print the metrics
    print("Compression Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
