import cv2
import heapq
import os
from collections import defaultdict
import numpy as np
import time
from math import log2

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    freq = defaultdict(int)
    for pixel in data:
        freq[pixel] += 1
    heap = [HuffmanNode(char, f) for char, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    return heap[0]

def generate_huffman_codes(node, current_code="", codes=None):
    if codes is None:
        codes = {}
    if node:
        if node.char is not None:
            codes[node.char] = current_code
        generate_huffman_codes(node.left, current_code + '0', codes)
        generate_huffman_codes(node.right, current_code + '1', codes)
    return codes

def encode_data(data, huffman_codes):
    return ''.join(huffman_codes[pixel] for pixel in data)

def write_binary_file(encoded_data, filename, huffman_codes, image_shape):
    with open(filename, 'wb') as f:
        height, width, channels = image_shape
        f.write(width.to_bytes(4, 'big'))
        f.write(height.to_bytes(4, 'big'))
        f.write(channels.to_bytes(1, 'big'))
        f.write(len(huffman_codes).to_bytes(4, 'big'))
        for pixel, code in huffman_codes.items():
            f.write(int(pixel).to_bytes(1, 'big'))
            f.write(len(code).to_bytes(1, 'big'))
            f.write(int(code, 2).to_bytes((len(code) + 7) // 8, 'big'))
        padding = 8 - len(encoded_data) % 8
        encoded_data += '0' * padding
        for i in range(0, len(encoded_data), 8):
            f.write(bytes([int(encoded_data[i:i+8], 2)]))

def process_image(input_image_path, output_file_path):
    image = cv2.imread(input_image_path)
    original_size = os.path.getsize(input_image_path)

    # Start compression
    start_compress = time.time()
    pixel_data = image.flatten()

    # Build Huffman tree and generate codes
    huffman_tree = build_huffman_tree(pixel_data)
    huffman_codes = generate_huffman_codes(huffman_tree)

    # Encode the pixel data
    encoded_data = encode_data(pixel_data, huffman_codes)

    # Write the encoded data to file
    write_binary_file(encoded_data, output_file_path, huffman_codes, image.shape)
    end_compress = time.time()
    compressed_size = os.path.getsize(output_file_path)
    time_taken_compress = end_compress - start_compress

    # Calculate entropy and redundancy
    unique, counts = np.unique(pixel_data, return_counts=True)
    probabilities = counts / len(pixel_data)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    avg_bits_per_symbol = sum(len(huffman_codes[p]) * counts[i] for i, p in enumerate(unique)) / len(pixel_data)
    redundancy = avg_bits_per_symbol - entropy

    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": original_size / compressed_size,
        "time_taken_compress": time_taken_compress,
        "entropy": entropy,
        "redundancy": redundancy
    }
