from src.utils import apply_idct_and_dequantize, load_from_file, inverse_zigzag_scan, run_length_decode,load_from_binary_file
from src.huffman import huffman_decode
import cv2
import struct
import numpy as np

def load_from_binary_file(input_file):
    with open(input_file, "rb") as f:
        # Read shape (height, width)
        shape = struct.unpack('ii', f.read(8))
        # Read quality factor
        quality_factor = struct.unpack('i', f.read(4))[0]
        # Read length of encoded binary data
        encoded_length = struct.unpack('i', f.read(4))[0]
        # Read encoded binary data
        byte_array = f.read((encoded_length + 7) // 8)
        encoded_data = bin(int.from_bytes(byte_array, byteorder='big'))[2:].zfill(encoded_length)
        # Read Huffman codes
        huffman_codes_length = struct.unpack('i', f.read(4))[0]
        huffman_codes_bytes = f.read(huffman_codes_length)
        huffman_codes = eval(huffman_codes_bytes.decode())
    return shape, encoded_data, huffman_codes, quality_factor

def scale_quantization_matrix(quant_matrix, quality_factor):
    """
    Scale the quantization matrix based on the quality factor.
    """
    if quality_factor < 1:
        quality_factor = 1
    if quality_factor > 100:
        quality_factor = 100

    scale = 50.0 / quality_factor if quality_factor < 50 else 2 - (quality_factor / 50.0)
    scaled_matrix = np.floor((quant_matrix * scale) + 0.5)
    return np.clip(scaled_matrix, 1, None)  # Ensure no element is zero or negative

def decoder_main(input_file, output_image_path, input_image_path):
    # Load data from the compressed file
    shape, encoded_data, huffman_codes, quality_factor = load_from_binary_file(input_file)
    h_padded, w_padded = shape  # The padded dimensions saved during encoding
    print(f'I am in decoder shape == {shape}')
    # Huffman decoding
    decoded_data = huffman_decode(encoded_data, huffman_codes)
    
    # Parse RLE and reverse zigzag scanning
    block_size = 8
    blocks = []
    current_block = []
    for item in decoded_data:
        if item == (-999,):  # End-of-block marker
            if current_block:
                rle_decoded = run_length_decode(current_block)
                block = inverse_zigzag_scan(rle_decoded, block_size)
                blocks.append(block)
                current_block = []
        else:
            current_block.append(item)
    
    # Reconstruct the quantized image
    quantized_image = np.zeros((h_padded, w_padded), dtype=np.float32)
    idx = 0
    for i in range(0, h_padded, block_size):
        for j in range(0, w_padded, block_size):
            quantized_image[i:i+block_size, j:j+block_size] = blocks[idx]
            idx += 1
    
    # Define the base quantization matrix
    base_quant_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    
    # Scale the quantization matrix
    scaled_quant_matrix = scale_quantization_matrix(base_quant_matrix, quality_factor)
    
    # Dequantize and apply IDCT
    reconstructed_image = apply_idct_and_dequantize(quantized_image, scaled_quant_matrix)

    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape    
    # Crop the image to original dimensions
    reconstructed_image_cropped = reconstructed_image[:h, :w]
    # print()
    # Save the reconstructed image
    cv2.imwrite(output_image_path, reconstructed_image_cropped)
