import os
import numpy as np
from PIL import Image
from src.dct import split_image_into_blocks, apply_dct_to_blocks, quantize_block, dequantize_block, standard_quantization_matrix
from src.huffman import build_huffman_tree, huffman_codes
from src.metrics import calculate_rmse, calculate_bpp

def compress_image(filename, output_folder="data/output", quality_factor=1.0):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load and process image
    image = Image.open(filename)
    blocks, h, w = split_image_into_blocks(image)

    # Apply DCT
    dct_blocks = apply_dct_to_blocks(blocks)

    # Quantize blocks
    quantized_blocks = [quantize_block(block, standard_quantization_matrix * quality_factor) for block in dct_blocks]

    # Flatten the quantized data for Huffman encoding
    flat_data = [int(item) for block in quantized_blocks for row in block for item in row]
    
    # Build Huffman Tree and encode
    root = build_huffman_tree(flat_data)
    codes = huffman_codes(root)
    encoded_data = ''.join(codes[value] for value in flat_data)

    # Save compressed data to a binary file
    compressed_filename = os.path.join(output_folder, "compressed_data.bin")
    with open(compressed_filename, "wb") as f:
        f.write(encoded_data.encode())  # Write encoded data as binary
    reconstructed_blocks = [dequantize_block(block, standard_quantization_matrix * quality_factor) for block in quantized_blocks]
    reconstructed_image = Image.fromarray(reconstruct_image_from_blocks(reconstructed_blocks, h, w))
    reconstructed_image_filename = os.path.join(output_folder, "reconstructed_image.png")
    reconstructed_image.save(reconstructed_image_filename)
    rmse = calculate_rmse(np.array(image), np.array(reconstructed_image))
    bpp = calculate_bpp(encoded_data, h, w)

    print(f"Compression completed:\n - Compressed data saved to {compressed_filename}\n - Reconstructed image saved to {reconstructed_image_filename}")
    print(f"RMSE: {rmse:.4f}, BPP: {bpp:.4f}")

def reconstruct_image_from_blocks(blocks, h, w, block_size=8):
    reconstructed_image = np.zeros((h, w), dtype=np.float32)
    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            reconstructed_image[i:i+block_size, j:j+block_size] = blocks[idx]
            idx += 1
    return np.clip(reconstructed_image, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    compress_image("data/input/sample.jpg", quality_factor=1.0)
