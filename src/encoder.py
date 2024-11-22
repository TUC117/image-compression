from src.utils import apply_dct_and_quantize, save_to_file, zigzag_scan, run_length_encode,save_to_binary_file
from src.huffman import huffman_encode
import cv2
import numpy as np
import struct 

def save_to_binary_file(output_file, shape, encoded_data, huffman_codes, quality_factor, is_colored):
    if not is_colored:
        # Convert encoded data from binary string to bytes
        byte_array = int(encoded_data, 2).to_bytes((len(encoded_data) + 7) // 8, byteorder='big')

        # Save to binary file
        with open(output_file, "wb") as f:
            # Save shapxe (height, width), quality factor, and length of encoded data
            f.write(struct.pack('ii', shape[0], shape[1]))  # Save height and width as integers
            f.write(struct.pack('i', quality_factor))  # Save quality factor as an integer
            f.write(struct.pack('i', len(encoded_data)))  # Save the length of the binary string
            f.write(byte_array)  # Save the encoded binary data
            # Save Huffman codes as a serialized dictionary
            huffman_codes_bytes = str(huffman_codes).encode()
            f.write(struct.pack('i', len(huffman_codes_bytes)))
            f.write(huffman_codes_bytes)
    else:
        print("I am color save binary")
        with open(output_file, "wb") as f:
            # Save the shape and quality factor
            f.write(struct.pack('ii', shape[0], shape[1]))  # Save height and width
            f.write(struct.pack('i', quality_factor))  # Save quality factor

            # Save encoded data and Huffman codes for each channel
            for encoded_data, huffman_codes in zip(encoded_data, huffman_codes):
                # Convert binary string to bytes
                byte_array = int(encoded_data, 2).to_bytes((len(encoded_data) + 7) // 8, byteorder='big')
                
                # Write the length of the encoded data
                f.write(struct.pack('i', len(encoded_data)))  
                
                # Write the encoded binary data
                f.write(byte_array)
                
                # Write Huffman codes as a serialized dictionary
                huffman_codes_bytes = str(huffman_codes).encode('utf-8')
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

    quality_factor = max(1, min(quality_factor, 100))

    # Compute the scale based on the quality factor
    scale = 50.0 / quality_factor if quality_factor < 50 else 2 - (quality_factor / 50.0)
    # scale = 50.0 / quality_factor

    # Scale the quantization matrix and round values
    scaled_matrix = np.floor((quant_matrix * scale) + 0.5)

    # Normalize by subtracting 128
    normalized_matrix = scaled_matrix - 128
    # normalized_matrix = scaled_matrix
#     normalized_matrix[scaled_matrix > 255] = 255
    # normalized_matrix[scaled_matrix <= 0] = 1    

    # Clip values to ensure a minimum of 1 (adjusted after normalization)
    normalized_matrix = np.clip(normalized_matrix, 1, None)

    return normalized_matrix

def is_colored_image(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Check if all channels are identical
        b, g, r = cv2.split(image)
        if (b == g).all() and (g == r).all():
            return False 
        else:
            return True
    return False  
    
def encoder_main(input_image_path, output_file, quality_factor):
    # Load the grayscale image
    
    image = cv2.imread(input_image_path)
    is_colored = is_colored_image(image)
    
    if not is_colored:
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
        
    else:
        print("YESS I AM COLOR encoding main")
        image = cv2.imread(input_image_path)
        if image is None:
            raise ValueError("Image not found or invalid format.")

        # Convert image to YCbCr for better compression performance
        ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(ycbcr_image)

        h, w = channels[0].shape
        padded_channels = []
        encoded_data_per_channel = []
        huffman_codes_per_channel = []

        for channel_idx, channel in enumerate(channels):  # Use 'channel_idx' for clarity
            # Pad each channel to ensure dimensions are divisible by 8
            print(f'Original size - {(h, w)} ')
            pad_h = (8 - (h % 8)) % 8
            pad_w = (8 - (w % 8)) % 8
            padded_channel = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            padded_channels.append(padded_channel)

            # Quantization matrix (use different matrices for Y and CbCr if needed)
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
            quant_matrix = base_quant_matrix if channel_idx == 0 else base_quant_matrix * 2

            # Apply DCT and quantization
            quantized_image = apply_dct_and_quantize(padded_channel, quant_matrix)

            # Perform RLE and zigzag
            encoded_blocks = []
            h_padded, w_padded = quantized_image.shape
            block_size = 8
            for i in range(0, h_padded, block_size):
                for j in range(0, w_padded, block_size):
                    block = quantized_image[i:i + block_size, j:j + block_size]
                    zigzag = zigzag_scan(block)
                    rle = run_length_encode(zigzag)
                    encoded_blocks.append(rle)
                    encoded_blocks.append([(-999,)])  # End-of-block marker

            # Flatten and prepare data for Huffman encoding
            flat_data = [item for block in encoded_blocks for item in block]
            encoded_data, huffman_codes = huffman_encode(flat_data)
            encoded_data_per_channel.append(encoded_data)
            huffman_codes_per_channel.append(huffman_codes)

        # Save to file
        save_to_binary_file(output_file, (h + pad_h, w + pad_w), encoded_data_per_channel, huffman_codes_per_channel, quality_factor,is_colored)
