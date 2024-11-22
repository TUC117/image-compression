from src.utils import apply_idct_and_dequantize, load_from_file, inverse_zigzag_scan, run_length_decode,load_from_binary_file
from src.huffman import huffman_decode
import cv2
import json
import struct
import numpy as np

def load_from_binary_file(input_file, is_colored):
    if not is_colored:
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
    else:
        print("COLOR IMAGE PART _ loading")
        with open(input_file, "rb") as f:
            # Read shape (height, width)
            shape = struct.unpack('ii', f.read(8))
            # Read quality factor
            quality_factor = struct.unpack('i', f.read(4))[0]

            encoded_data = []
            huffman_codes = []

            # Read encoded data and Huffman codes for each channel
            for _ in range(3):  # Iterate for each channel (R, G, B)
                # Read the length of the encoded data (in bits)
                encoded_length = struct.unpack('i', f.read(4))[0]
                
                # Read encoded binary data
                byte_array = f.read((encoded_length + 7) // 8)
                channel_encoded_data = bin(int.from_bytes(byte_array, byteorder='big'))[2:].zfill(encoded_length)
                encoded_data.append(channel_encoded_data)

                # Read Huffman codes
                huffman_codes_length = struct.unpack('i', f.read(4))[0]
                huffman_codes_bytes = f.read(huffman_codes_length)
                channel_huffman_codes = eval(huffman_codes_bytes.decode())
                huffman_codes.append(channel_huffman_codes)

        return shape, encoded_data, huffman_codes, quality_factor


def scale_quantization_matrix(quant_matrix, quality_factor):
    """
    Scale the quantization matrix based on the quality factor.
    """
    if quality_factor < 1:
        quality_factor = 1
    if quality_factor > 100:
        quality_factor = 100

    quality_factor = max(1, min(quality_factor, 100))

    # Compute the scale based on the quality factor
    scale = 50.0 / quality_factor if quality_factor < 50 else 2 - (quality_factor / 50.0)
    # scale = 50.0 / quality_factor

    # Scale the quantization matrix and round values
    scaled_matrix = np.floor((quant_matrix * scale) + 0.5)

    # Normalize by subtracting 128
    normalized_matrix = scaled_matrix - 128

    # Clip values to ensure a minimum of 1 (adjusted after normalization)
    normalized_matrix = np.clip(normalized_matrix, 1, None)

    return normalized_matrix


def decoder_main(input_file, output_image_path, input_image_path):
    # Load data from the compressed file
    is_colored = True
    if not is_colored:
        shape, encoded_data, huffman_codes, quality_factor = load_from_binary_file(input_file, is_colored)
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
    else:
        print("I am color decoding main")
        shape, encoded_data_per_channel, huffman_codes_per_channel, quality_factor = load_from_binary_file(input_file, is_colored)
        h_padded, w_padded = shape
        block_size = 8
        
        print()
        print("After Loading...")
        print((h_padded, w_padded))
        print(len(encoded_data_per_channel), len(encoded_data_per_channel[0]))
        print(len(huffman_codes_per_channel), len(huffman_codes_per_channel[0]))
        print(quality_factor)
        
        reconstructed_channels = []

        for encoded_data, huffman_codes in zip(encoded_data_per_channel, huffman_codes_per_channel):
            # Huffman decoding
            decoded_data = huffman_decode(encoded_data, huffman_codes)

            # RLE and inverse zigzag
            blocks = []
            block_size = 8
            current_block = []
            for item in decoded_data:
                if item == (-999,):  # End-of-block marker
                    if current_block:
                        rle_decoded = run_length_decode(current_block)
                        if len(rle_decoded) < 64:  # Pad if necessary
                            rle_decoded.extend([0] * (64 - len(rle_decoded)))
                        block = inverse_zigzag_scan(rle_decoded, block_size)
                        blocks.append(block)
                        current_block = []
                else:
                    current_block.append(item)

            # Reconstruct quantized image
            quantized_image = np.zeros((h_padded, w_padded), dtype=np.float32)
            print("PRINTINTTTTT")
            print( len(current_block))
            idx = 0
            for i in range(0, h_padded, block_size):
                # print(len(blocks[idx]))
                for j in range(0,   w_padded, block_size):
                    # if idx < len(blocks):
                    quantized_image[i:i + block_size, j:j + block_size] = blocks[idx]
                    idx += 1

            # Dequantize and apply IDCT
            base_quant_matrix = np.array([
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99],
            ])
            # quant_matrix = base_quant_matrix if len(reconstructed_channels) == 0 else base_quant_matrix * 2
            scaled_quant_matrix = scale_quantization_matrix(base_quant_matrix, quality_factor)
            reconstructed_channel = apply_idct_and_dequantize(quantized_image, scaled_quant_matrix)
            reconstructed_channels.append(reconstructed_channel)

        input_image = cv2.imread(input_image_path)
        ycbcr_image_input = cv2.cvtColor(input_image, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(ycbcr_image_input)
        
        h, w = channels[0].shape

        # Merge channels and convert back to BGR
        ycbcr_image = cv2.merge(reconstructed_channels)
        reconstructed_image = cv2.cvtColor(ycbcr_image, cv2.COLOR_YCrCb2BGR)

        # Crop to remove padding and restore original dimensions
        reconstructed_image_cropped = reconstructed_image[:h, :w]

        # Save the reconstructed image
        cv2.imwrite(output_image_path, reconstructed_image_cropped)
