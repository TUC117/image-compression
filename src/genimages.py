import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to calculate RMSE
def calculate_rmse(original, compressed):
    return np.sqrt(np.mean((original - compressed) ** 2))

# Function to calculate BPP
def calculate_bpp(image_size_bits, width, height):
    return image_size_bits / (width * height)

def plot_main(original_image_path, compressed_images_dir):
    # Read the original image
    original = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Get dimensions of the original image
    height, width = original.shape

    # Lists to store results for this specific image
    rmse_values = []
    bpp_values = []
    image_name = original_image_path.split('/')[-1]
    # Process all compressed versions of the specific image
    compressed_image_paths = sorted(os.listdir(compressed_images_dir))  # Adjust naming pattern if needed
    for compressed_path in compressed_image_paths:
        # Read the compressed image
        if compressed_path.endswith('.png'):
            print(compressed_path)
            real_path = os.path.join(compressed_images_dir, compressed_path)
            compressed = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
            
            # Ensure dimensions match
            if original.shape != compressed.shape:
                raise ValueError(f"Shape mismatch: {original.shape} vs {compressed.shape}")
            
            # Calculate RMSE
            rmse = calculate_rmse(original.astype(np.float64), compressed.astype(np.float64))
            rmse_values.append(rmse)
            
            # Get compressed size in bits
            # compressed_size_bits = len(open(real_path, "rb").read()) * 8
            compressed_size_bits = os.path.getsize(real_path) * 8
            bpp = calculate_bpp(compressed_size_bits, width, height)
            
            bpp_original = calculate_bpp(os.path.getsize(original_image_path) * 8 , width, height)
            print(f"Original BPP - {bpp_original}")
            bpp_values.append(bpp)
        else:
            raise NameError(f"mismatchh of format for {compressed_path}")
    
    # Plot RMSE vs. BPP for this specific image
    plt.figure(figsize=(8, 6))

    # Add scatter points with different markers for each quality factor
    for bpp, rmse, compressed_path in zip(bpp_values, rmse_values, compressed_image_paths):
        quality_factor = compressed_path.split('-')[-1].split('.')[0]  # Extract the number after the '-'
        plt.scatter(bpp, rmse, label=f"QF: {quality_factor}")

    plt.plot(bpp_values, rmse_values, linestyle="--", color="blue", label="RMSE vs. BPP")

    # Customize plot
    plt.title(f"RMSE vs. BPP for {image_name.split('.')[0]} Compression")
    plt.xlabel("Bits Per Pixel (BPP)")
    plt.ylabel("Root Mean Squared Error (RMSE)")
    plt.legend(title="Quality Factors", loc="upper right")  # Legend title and position
    plt.grid(True)
    plt.savefig(os.path.join('plots/', f"{image_name}"))
