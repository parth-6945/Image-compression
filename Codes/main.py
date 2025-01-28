import os
import csv
from huffman_image_compression_only import process_image as huffman_process
from Limpel_ziv import LZW  # Assuming your LZW code is in Limpel_ziv.py
from rle import RLE  # Assuming your RLE code is in rle_image_compression.py
from deflate import compress_image_with_deflate as deflate_compress  # Correct import for Deflate
from predictive import predictive_coding_compression, predictive_coding_decompression  # Import your predictive coding functions
from jpeg2000 import jpeg2000_compression, jpeg2000_decompression  # Import JPEG 2000 compression functions
from webp import webp_compression, webp_decompression  # Import WebP compression functions
from png import png_compression  # Import PNG compression functions


def process_images_in_folder(input_folder, output_folder, csv_file_path):
    os.makedirs(output_folder, exist_ok=True)

    valid_extensions = (".bmp", ".dng", ".jpg", ".png")  # Include more extensions as needed
    write_header = not os.path.exists(csv_file_path)

    with open(csv_file_path, mode="a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        if write_header:
            csv_writer.writerow([
                "Filename", "Original Size (bytes)", "Compressed Size (bytes)",
                "Compression Ratio", "Time to Compress (seconds)", "Entropy", "Redundancy", "Coding Technique"
            ])

        for filename in os.listdir(input_folder):
            input_image_path = os.path.join(input_folder, filename)
            if os.path.isfile(input_image_path) and filename.lower().endswith(valid_extensions):
                print(f"Processing: {filename}")
                compressed_file_path_huffman = os.path.join(output_folder, filename + ".huf")
                compressed_file_path_lzw = os.path.join(output_folder, filename + ".lzw")
                compressed_file_path_rle = os.path.join(output_folder, filename + ".rle")
                compressed_file_path_predictive = os.path.join(output_folder, filename + ".predictive.gz")
                compressed_file_path_jpeg2000 = os.path.join(output_folder, filename + ".jp2")
                compressed_file_path_webp = os.path.join(output_folder, filename + ".webp")
                compressed_file_path_png = os.path.join(output_folder, filename + ".png")

                # Huffman compression
                try:
                    huffman_result = huffman_process(input_image_path, compressed_file_path_huffman)
                    csv_writer.writerow([
                        filename, huffman_result["original_size"], huffman_result["compressed_size"],
                        huffman_result["compression_ratio"], huffman_result["time_taken_compress"],
                        huffman_result["entropy"], huffman_result["redundancy"], "huffman"
                    ])
                except Exception as e:
                    print(f"Error processing {filename} with Huffman: {e}")

                # LZW compression
                try:
                    lzw = LZW(input_image_path)
                    lzw_result = lzw.compress()
                    csv_writer.writerow([
                        filename, lzw_result["original_size"], lzw_result["compressed_size"],
                        lzw_result["compression_ratio"], lzw_result["time_taken_compress"],
                        lzw_result["entropy"], lzw_result["redundancy"], "lzw"
                    ])
                except Exception as e:
                    print(f"Error processing {filename} with LZW: {e}")

                # RLE compression
                try:
                    rle = RLE(input_image_path)
                    rle_result = rle.compress()
                    csv_writer.writerow([
                        filename, rle_result["original_size"], rle_result["compressed_size"],
                        rle_result["compression_ratio"], rle_result["time_taken_compress"],
                        rle_result["entropy"], rle_result["redundancy"], "rle"
                    ])
                except Exception as e:
                    print(f"Error processing {filename} with RLE: {e}")

                # Deflate compression
                try:
                    deflate_result = deflate_compress(input_image_path)
                    csv_writer.writerow([
                        filename, deflate_result["original_size"], deflate_result["compressed_size"],
                        deflate_result["compression_ratio"], deflate_result["time_taken_compress"],
                        deflate_result["entropy"], deflate_result["redundancy"], "deflate"
                    ])
                except Exception as e:
                    print(f"Error processing {filename} with Deflate: {e}")

                # Predictive Coding compression
                try:
                    predictive_result = predictive_coding_compression(input_image_path, compressed_file_path_predictive)
                    csv_writer.writerow([
                        filename, predictive_result["original_size"], predictive_result["compressed_size"],
                        predictive_result["compression_ratio"], predictive_result["time_taken"],
                        predictive_result["entropy"], predictive_result["redundancy"], "predictive_coding"
                    ])
                except Exception as e:
                    print(f"Error processing {filename} with Predictive Coding: {e}")

                # JPEG 2000 compression
                try:
                    jpeg2000_result = jpeg2000_compression(input_image_path, compressed_file_path_jpeg2000)
                    csv_writer.writerow([
                        filename, jpeg2000_result["original_size"], jpeg2000_result["compressed_size"],
                        jpeg2000_result["compression_ratio"], jpeg2000_result["time_taken"],
                        jpeg2000_result["entropy"], jpeg2000_result["redundancy"], "jpeg2000"
                    ])
                except Exception as e:
                    print(f"Error processing {filename} with JPEG 2000: {e}")

                # WebP compression
                try:
                    webp_result = webp_compression(input_image_path, compressed_file_path_webp)
                    csv_writer.writerow([
                        filename, webp_result["original_size"], webp_result["compressed_size"],
                        webp_result["compression_ratio"], webp_result["time_taken"],
                        webp_result["entropy"], webp_result["redundancy"], "webp"
                    ])
                except Exception as e:
                    print(f"Error processing {filename} with WebP: {e}")

                # PNG compression
                try:
                    png_result = png_compression(input_image_path, compressed_file_path_png)
                    csv_writer.writerow([
                        filename, png_result["original_size"], png_result["compressed_size"],
                        png_result["compression_ratio"], png_result["time_taken"],
                        png_result["entropy"], png_result["redundancy"], "png"
                    ])
                except Exception as e:
                    print(f"Error processing {filename} with PNG: {e}")


if __name__ == "__main__":
    input_folder = r"Compression\Images"
    output_folder = r"Compression\Compressed"
    csv_file_path = r"Compression\compression_results.csv"

    process_images_in_folder(input_folder, output_folder, csv_file_path)
