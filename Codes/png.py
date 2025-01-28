from PIL import Image
import os
import time
import numpy as np
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


def png_compression(input_image_path, output_image_path):
    start_time = time.time()

    # Open the image using Pillow
    image = Image.open(input_image_path)
    
    # Save the image as PNG with lossless compression
    image.save(output_image_path, format="PNG", compress_level=9)

    end_time = time.time()

    # Get the sizes
    original_size = os.path.getsize(input_image_path)
    compressed_size = os.path.getsize(output_image_path)
    compression_ratio = original_size / compressed_size
    time_taken = end_time - start_time

    # Load the compressed image to calculate entropy and redundancy
    compressed_image = np.array(image)

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

if __name__ == "__main__":
    input_image_path = "Wallpaper_1.jpg"  # Replace with your image path
    output_image_path = "output_image.png"

    compression_result = png_compression(input_image_path, output_image_path)
    print("PNG Compression Results:", compression_result)
