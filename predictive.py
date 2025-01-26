import numpy as np
import cv2
import gzip
import os
import time
from collections import Counter


def predictive_coding_compression(image_path, output_path):
    start_time = time.time()

    # Read the image in RGB (BGR format in OpenCV)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Get image dimensions
    height, width, channels = image.shape

    # Initialize the prediction error for each channel with np.int16 to avoid overflow
    prediction_error = np.zeros((height, width, channels), dtype=np.int16)

    # Predictive coding using the Median Edge Detector (MED) for each channel
    for i in range(height):
        for j in range(width):
            if i == 0 or j == 0:  # Top row or left column
                predicted_pixel = [0, 0, 0]  # For R, G, B
            else:
                left = image[i, j - 1].astype(np.int16)  # Use int16 for intermediate calculations
                top = image[i - 1, j].astype(np.int16)
                top_left = image[i - 1, j - 1].astype(np.int16)

                # MED predictor for each channel
                predicted_pixel = []
                for c in range(3):  # Loop over the RGB channels
                    if top_left[c] >= max(left[c], top[c]):
                        predicted_pixel.append(min(left[c], top[c]))
                    elif top_left[c] <= min(left[c], top[c]):
                        predicted_pixel.append(max(left[c], top[c]))
                    else:
                        predicted_pixel.append(left[c] + top[c] - top_left[c])

            # Calculate prediction error for each channel (store in int16 to avoid overflow)
            prediction_error[i, j] = image[i, j] - predicted_pixel

    # Flatten the prediction error to calculate entropy
    flat_prediction_error = prediction_error.flatten()

    # Calculate probabilities of unique values
    value_counts = Counter(flat_prediction_error)
    total_values = sum(value_counts.values())
    probabilities = [count / total_values for count in value_counts.values()]

    # Calculate entropy
    entropy = -sum(p * np.log2(p) for p in probabilities)

    # Calculate maximum entropy for the data type
    max_entropy = np.log2(2 ** 16)  # 16 bits (int16)

    # Calculate redundancy
    redundancy = max_entropy - entropy

    # Compress the prediction error using gzip
    with gzip.open(output_path, "wb") as f:
        f.write(prediction_error.astype(np.int16).tobytes())

    end_time = time.time()

    original_size = os.path.getsize(image_path)
    compressed_size = os.path.getsize(output_path)
    compression_ratio = original_size / compressed_size
    time_taken_compress = end_time - start_time

    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": compression_ratio,
        "time_taken": time_taken_compress,
        "entropy": entropy,
        "redundancy": redundancy,
    }


def predictive_coding_decompression(compressed_path, image_shape, output_image_path):
    # Load the compressed prediction error
    with gzip.open(compressed_path, "rb") as f:
        prediction_error = np.frombuffer(f.read(), dtype=np.int16).reshape(image_shape)

    # Initialize the reconstructed image
    height, width, channels = image_shape
    reconstructed_image = np.zeros((height, width, channels), dtype=np.uint8)

    # Reconstruct the image using the prediction error for each channel
    for i in range(height):
        for j in range(width):
            if i == 0 or j == 0:  # Top row or left column
                predicted_pixel = [0, 0, 0]  # For R, G, B
            else:
                left = reconstructed_image[i, j - 1].astype(np.int16)
                top = reconstructed_image[i - 1, j].astype(np.int16)
                top_left = reconstructed_image[i - 1, j - 1].astype(np.int16)

                # MED predictor for each channel
                predicted_pixel = []
                for c in range(3):  # Loop over the RGB channels
                    if top_left[c] >= max(left[c], top[c]):
                        predicted_pixel.append(min(left[c], top[c]))
                    elif top_left[c] <= min(left[c], top[c]):
                        predicted_pixel.append(max(left[c], top[c]))
                    else:
                        predicted_pixel.append(left[c] + top[c] - top_left[c])

            # Reconstruct the pixel value for each channel
            reconstructed_pixel = np.array(predicted_pixel) + prediction_error[i, j]
            reconstructed_image[i, j] = np.clip(reconstructed_pixel, 0, 255)

    # Save the reconstructed image
    cv2.imwrite(output_image_path, reconstructed_image)

    return reconstructed_image


if __name__ == "__main__":
    input_image_path = "Wallpaper_1.jpg"  # Replace with your image path
    compressed_path = "compressed_prediction_error.gz"
    reconstructed_image_path = "reconstructed_image.png"

    # Compression
    compression_result = predictive_coding_compression(input_image_path, compressed_path)
    print("Compression Results:", compression_result)

    # Get the shape of the original image
    original_image = cv2.imread(input_image_path)
    image_shape = original_image.shape

    # Decompression
    reconstructed_image = predictive_coding_decompression(
        compressed_path, image_shape, reconstructed_image_path
    )

    # Verify
    if np.array_equal(original_image, reconstructed_image):
        print("Decompression verified: Original and reconstructed images match!")
    else:
        print("Decompression failed: Original and reconstructed images do not match.")
