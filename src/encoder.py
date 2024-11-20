from src.utils import apply_dct_and_quantize, save_to_file, zigzag_scan, run_length_encode,save_to_binary_file
from src.huffman import huffman_encode
import cv2
import numpy as np
import struct 

def save_to_binary_file(output_file, shape, encoded_data, huffman_codes, quality_factor):
    # Convert encoded data from binary string to bytes
    byte_array = int(encoded_data, 2).to_bytes((len(encoded_data) + 7) // 8, byteorder='big')

    # Save to binary file
    with open(output_file, "wb") as f:
        # Save shape (height, width), quality factor, and length of encoded data
        f.write(struct.pack('ii', shape[0], shape[1]))  # Save height and width as integers
        f.write(struct.pack('i', quality_factor))  # Save quality factor as an integer
        f.write(struct.pack('i', len(encoded_data)))  # Save the length of the binary string
        f.write(byte_array)  # Save the encoded binary data
        # Save Huffman codes as a serialized dictionary
        huffman_codes_bytes = str(huffman_codes).encode()
        f.write(struct.pack('i', len(huffman_codes_bytes)))
        f.write(huffman_codes_bytes)

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

def encoder_main(input_image_path, output_file, quality_factor):
    # Load the grayscale image
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    print(f'Input image has shape - {h}, {w}')  
    
    # Calculate padding
    pad_h = (8 - (h % 8)) % 8  # Padding needed for height
    pad_w = (8 - (w % 8)) % 8  # Padding needed for width
    
    # Pad the image with zeros
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    h_padded, w_padded = padded_image.shape
    # print(f'Padded image has shape - {h_padded}, {w_padded}')
    
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
    
    # Apply DCT and quantization
    quantized_image = apply_dct_and_quantize(padded_image, scaled_quant_matrix)
    
    # Perform RLE and zigzag
    block_size = 8
    h_blocks = h_padded // block_size
    w_blocks = w_padded // block_size
    encoded_blocks = []
    
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = quantized_image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            zigzag = zigzag_scan(block)
            rle = run_length_encode(zigzag)
            encoded_blocks.append(rle)
            encoded_blocks.append([(-999,)])  # End-of-block marker
    
    # Flatten and prepare data for Huffman encoding
    flat_data = [item for block in encoded_blocks for item in block]
    encoded_data, huffman_codes = huffman_encode(flat_data)
    
    # Save to file
    save_to_binary_file(output_file, (h_padded, w_padded), encoded_data, huffman_codes, quality_factor)
