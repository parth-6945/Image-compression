import os

directory = r"C:\Users\parth\projects\Compression\Images"

for filename in os.listdir(directory):
    if "×" in filename:
        new_name = filename.replace("×", "x")
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
        print(f"Renamed: {filename} -> {new_name}")
