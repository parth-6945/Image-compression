import os
import numpy as np
from PIL import Image
import time


class RLE:
    def __init__(self, path):
        self.path = path
        print(f"Initialized RLE object for file: {self.path}")

    ''''''
    ''' --------------------- Compression of the Image --------------------- '''
    ''''''

    def compress(self):
        print("Starting image compression using RLE...")
        start_time = time.time()
        self.initCompress()
        compressedColors = []
        print("Compressing Red channel...")
        compressedColors.append(self.compressColor(self.red))
        print("Compressing Green channel...")
        compressedColors.append(self.compressColor(self.green))
        print("Compressing Blue channel...")
        compressedColors.append(self.compressColor(self.blue))
        print("Image compression complete. Writing compressed data to file...")

        filesplit = str(os.path.basename(self.path)).split('.')
        filename = filesplit[0] + 'Compressed.rle'
        savingDirectory = os.path.join(os.getcwd(), 'Compressed')
        os.makedirs(savingDirectory, exist_ok=True)

        compressed_file_path = os.path.join(savingDirectory, filename)
        with open(compressed_file_path, 'w') as file:
            for color in compressedColors:
                for row in color:
                    file.write(row + "\n")

        time_taken_compress = time.time() - start_time
        original_size = os.path.getsize(self.path)
        compressed_size = os.path.getsize(compressed_file_path)
        compression_ratio = original_size / compressed_size

        entropy = self.calculate_entropy()
        redundancy = 1 - (entropy / (8 * self.image.mode.count("RGB")))

        print(f"Compressed image saved as {filename}")
        print(f"Compression Metrics:")
        print(f" - Original size: {original_size} bytes")
        print(f" - Compressed size: {compressed_size} bytes")
        print(f" - Compression ratio: {compression_ratio:.2f}")
        print(f" - Time taken: {time_taken_compress:.2f} seconds")
        print(f" - Entropy: {entropy:.2f}")
        print(f" - Redundancy: {redundancy:.2f}")
        return {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "time_taken_compress": time_taken_compress,
            "entropy": entropy,
            "redundancy": redundancy
        }

    def compressColor(self, colorList):
        compressedColor = []
        for currentRow in colorList:
            compressedRow = []
            currentValue = currentRow[0]
            count = 1

            for i in range(1, len(currentRow)):
                if currentRow[i] == currentValue:
                    count += 1
                else:
                    compressedRow.append(f"{currentValue},{count}")
                    currentValue = currentRow[i]
                    count = 1

            compressedRow.append(f"{currentValue},{count}")  # Append the last value-count pair
            compressedColor.append(" ".join(compressedRow))
        return compressedColor

    def calculate_entropy(self):
        pixel_data = np.array(self.image)
        unique, counts = np.unique(pixel_data, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    ''''''
    ''' --------------------- Decompression of the Image --------------------- '''
    ''''''

    def decompress(self):
        print("Starting image decompression using RLE...")
        image = []
        with open(self.path, 'r') as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    decodedRow = self.decompressRow(line.strip())
                    image.append(np.array(decodedRow))

        image = np.array(image)
        shapeTup = image.shape
        image = image.reshape((3, shapeTup[0] // 3, shapeTup[1]))
        self.saveImage(image)
        print("Decompression complete. Decompressed image saved.")

    def decompressRow(self, line):
        currentRow = line.split(" ")
        decodedRow = []

        for segment in currentRow:
            try:
                value, count = map(int, segment.split(","))
                decodedRow.extend([value] * count)
            except ValueError:
                continue

        return decodedRow

    ''''''
    ''' ---------------------- Class Helper Functions ---------------------- '''
    ''''''

    def initCompress(self):
        self.image = Image.open(self.path)
        self.height, self.width = self.image.size
        self.red, self.green, self.blue = self.processImage()

    def processImage(self):
        image = self.image.convert('RGB')
        red, green, blue = [], [], []
        pixel_values = list(image.getdata())
        iterator = 0
        for height_index in range(self.height):
            R, G, B = [], [], []
            for width_index in range(self.width):
                RGB = pixel_values[iterator]
                R.append(RGB[0])
                G.append(RGB[1])
                B.append(RGB[2])
                iterator += 1
            red.append(R)
            green.append(G)
            blue.append(B)
        return red, green, blue

    def saveImage(self, image):
        filesplit = str(os.path.basename(self.path)).split('Compressed.rle')
        filename = filesplit[0] + "Decompressed.tif"
        savingDirectory = os.path.join(os.getcwd(), 'Decompressed')
        os.makedirs(savingDirectory, exist_ok=True)

        imagelist, imagesize = self.makeImageData(image[0], image[1], image[2])
        imagenew = Image.new('RGB', imagesize)
        imagenew.putdata(imagelist)
        imagenew.save(os.path.join(savingDirectory, filename))

    def makeImageData(self, r, g, b):
        imagelist = []
        for i in range(len(r)):
            for j in range(len(r[0])):
                imagelist.append((r[i][j], g[i][j], b[i][j]))
        return imagelist, (len(r), len(r[0]))


# Main function to trigger the RLE compression/decompression process
def main():
    image_path = 'Wallpaper_1.jpg'  # Replace with your image file path

    # Create an instance of the RLE class
    rle = RLE(image_path)

    # Compress the image and capture metrics
    compression_result = rle.compress()

    # Decompress the image (use the compressed file path)
    rle.path = 'CompressedFiles/Wallpaper_1Compressed.rle'
    rle.decompress()


if __name__ == "__main__":
    main()
