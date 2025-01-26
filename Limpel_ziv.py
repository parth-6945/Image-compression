import os
import numpy as np
import time
from PIL import Image
import math


class LZW:
    def __init__(self, path):
        self.path = path
        self.compressionDictionary, self.compressionIndex = self.createCompressionDict()
        self.decompressionDictionary, self.decompressionIndex = self.createDecompressionDict()
        print(f"Initialized LZW object for file: {self.path}")

    ''''''
    ''' --------------------- Compression of the Image --------------------- '''
    ''''''

    def compress(self):
        start_time = time.time()
        print("Starting image compression...")
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
        filename = filesplit[0] + 'Compressed.lzw'
        savingDirectory = os.path.join(os.getcwd(), 'Compressed')
        if not os.path.isdir(savingDirectory):
            os.makedirs(savingDirectory)

        compressed_file_path = os.path.join(savingDirectory, filename)

        with open(compressed_file_path, 'wb') as file:  # Open file in binary mode
            for color in compressedColors:
                for row in color:
                    file.write(row.encode('utf-8'))  # Encode to bytes using utf-8
                    file.write(b"\n")  # Write newline as a byte

        # Calculate compression time
        time_taken_compress = time.time() - start_time
        original_size = os.path.getsize(self.path)
        compressed_size = os.path.getsize(compressed_file_path)
        compression_ratio = original_size / compressed_size

        # Calculate entropy and redundancy (approximation for now)
        entropy = self.calculate_entropy(compressed_file_path)
        redundancy = self.calculate_redundancy(entropy)

        return {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "time_taken_compress": time_taken_compress,
            "entropy": entropy,
            "redundancy": redundancy
        }

    def compressColor(self, colorList):
        print("Starting compression for a color channel...")
        compressedColor = []
        for currentRow in colorList:
            currentString = currentRow[0]
            compressedRow = ""
            for charIndex in range(1, len(currentRow)):
                currentChar = currentRow[charIndex]
                if currentString + currentChar in self.compressionDictionary:
                    currentString = currentString + currentChar
                else:
                    compressedRow = compressedRow + str(self.compressionDictionary[currentString]) + ","
                    self.compressionDictionary[currentString + currentChar] = self.compressionIndex
                    self.compressionIndex += 1
                    currentString = currentChar
            compressedRow = compressedRow + str(self.compressionDictionary[currentString])
            compressedColor.append(compressedRow)
        print("Color channel compression complete.")
        return compressedColor

    ''''''
    ''' ---------------------- Helper Methods for Entropy and Redundancy ---------------------- '''
    ''''''

    def calculate_entropy(self, file_path):
        print("Calculating entropy of the compressed file...")
        with open(file_path, 'rb') as f:
            data = f.read()

        # Calculate the frequency of each byte (0-255)
        byte_freq = {}
        for byte in data:
            if byte not in byte_freq:
                byte_freq[byte] = 0
            byte_freq[byte] += 1

        # Calculate entropy
        total_bytes = len(data)
        entropy = 0.0
        for freq in byte_freq.values():
            probability = freq / total_bytes
            entropy -= probability * math.log2(probability)

        return entropy

    def calculate_redundancy(self, entropy):
        M = 256  # Number of possible values for each byte (for 8-bit color)
        redundancy = 1 - (entropy / math.log2(M))
        return redundancy

    ''''''
    ''' --------------------- Decompression of the Image --------------------- '''
    ''''''

    def decompress(self):
        print("Starting image decompression...")
        image = []
        with open(self.path, 'rb') as file:  # Open file in binary mode
            for line in file:
                line = line.decode('utf-8')  # Decode the byte string back to a regular string
                decodedRow = self.decompressRow(line)
                image.append(np.array(decodedRow))
        image = np.array(image)
        shapeTup = image.shape
        image = image.reshape((3, shapeTup[0] // 3, shapeTup[1]))
        self.saveImage(image)
        print("Decompression complete.")

    def decompressRow(self, line):
        print("Starting decompression for a row...")
        currentRow = line.split(",")
        currentRow[-1] = currentRow[-1][:-1]
        decodedRow = ""
        word, entry = "", ""
        decodedRow = decodedRow + self.decompressionDictionary[int(currentRow[0])]
        word = self.decompressionDictionary[int(currentRow[0])]
        for i in range(1, len(currentRow)):
            new = int(currentRow[i])
            if new in self.decompressionDictionary:
                entry = self.decompressionDictionary[new]
                decodedRow += entry
                add = word + entry[0]
                word = entry
            else:
                entry = word + word[0]
                decodedRow += entry
                add = entry
                word = entry
            self.decompressionDictionary[self.decompressionIndex] = add
            self.decompressionIndex += 1
        newRow = decodedRow.split(',')
        decodedRow = [int(x) for x in newRow]
        print("Row decompression complete.")
        return decodedRow

    ''''''
    ''' ---------------------- Class Helper Functions ---------------------- '''
    ''''''

    def initCompress(self):
        print(f"Opening image file: {self.path}")
        self.image = Image.open(self.path)
        self.height, self.width = self.image.size
        self.red, self.green, self.blue = self.processImage()

    def processImage(self):
        print("Processing image to separate RGB channels...")
        image = self.image.convert('RGB')
        red, green, blue = [], [], []
        pixel_values = list(image.getdata())
        iterator = 0
        for height_index in range(self.height):
            R, G, B = "", "", ""
            for width_index in range(self.width):
                RGB = pixel_values[iterator]
                R = R + str(RGB[0]) + ","
                G = G + str(RGB[1]) + ","
                B = B + str(RGB[2]) + ","
                iterator += 1
            red.append(R[:-1])
            green.append(G[:-1])
            blue.append(B[:-1])
        print("Image processing complete.")
        return red, green, blue

    def saveImage(self, image):
        print("Saving decompressed image...")
        filesplit = str(os.path.basename(self.path)).split('Compressed.lzw')
        filename = filesplit[0] + "Decompressed.tif"
        savingDirectory = os.path.join(os.getcwd(), 'Decompressed')
        if not os.path.isdir(savingDirectory):
            os.makedirs(savingDirectory)
        imagelist, imagesize = self.makeImageData(image[0], image[1], image[2])
        imagenew = Image.new('RGB', imagesize)
        imagenew.putdata(imagelist)
        imagenew.save(os.path.join(savingDirectory, filename))
        print(f"Decompressed image saved as {filename}")

    def makeImageData(self, r, g, b):
        imagelist = []
        for i in range(len(r)):
            for j in range(len(r[0])):
                imagelist.append((r[i][j], g[i][j], b[i][j]))
        return imagelist, (len(r), len(r[0]))

    def createCompressionDict(self):
        print("Creating compression dictionary...")
        dictionary = {}
        for i in range(10):
            dictionary[str(i)] = i
        dictionary[','] = 10
        print("Compression dictionary created.")
        return dictionary, 11

    def createDecompressionDict(self):
        print("Creating decompression dictionary...")
        dictionary = {}
        for i in range(10):
            dictionary[i] = str(i)
        dictionary[10] = ','
        print("Decompression dictionary created.")
        return dictionary, 11


# Main function to trigger the LZW compression/decompression process
def main():
    # Specify the path to the image file
    image_path = 'Wallpaper_1.jpg'  # Replace with your image file path
    
    # Create an instance of the LZW class
    lzw = LZW(image_path)
    
    # To compress the image and get output in required format
    result = lzw.compress()
    print(result)

if __name__ == "__main__":
    main()
