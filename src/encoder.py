from src.utils import apply_dct_and_quantize, save_to_file, zigzag_scan, run_length_encode,save_to_binary_file
from src.huffman import huffman_encode
import cv2
import numpy as np
import struct 
import json

<<<<<<< HEAD
def save_to_binary_file(output_file, is_colored, shape, encoded_data, huffman_codes, quality_factor):
    if is_colored==0: 
=======
def save_to_binary_file(output_file, shape, encoded_data, huffman_codes, quality_factor, is_colored):
    is_colored = False
    if not is_colored: 
>>>>>>> b1699b4 (Final Commit)
        # Convert encoded data from binary string to bytes
        byte_array = int(encoded_data, 2).to_bytes((len(encoded_data) + 7) // 8, byteorder='big')
        # Save to binary file
        with open(output_file, "wb") as f:
            # Save shape (height, width), quality factor, and length of encoded data
            f.write(struct.pack('i', is_colored))
            f.write(struct.pack('ii', shape[0], shape[1]))  # Save height and width as integers
            f.write(struct.pack('i', quality_factor))  # Save quality factor as an integer
            f.write(struct.pack('i', len(encoded_data)))  # Save the length of the binary string
            f.write(byte_array)  # Save the encoded binary data
            # Save Huffman codes as a serialized dictionary
            huffman_codes_bytes = str(huffman_codes).encode()
            f.write(struct.pack('i', len(huffman_codes_bytes)))
            f.write(huffman_codes_bytes)
    else:
        with open(output_file, "wb") as f:
            # Save the shape and quality factor
            f.write(struct.pack('i', is_colored))
            f.write(struct.pack('ii', shape[0], shape[1]))  # Save height and width
            f.write(struct.pack('i', quality_factor))  # Save quality factor
            list_encoded_data = encoded_data
            list_huffman_codes = huffman_codes
            # Save encoded data and Huffman codes for each channel
            for encoded_data1, huffman_codes1 in zip(list_encoded_data, list_huffman_codes):
                # Convert binary string to bytes
                byte_array = int(encoded_data1, 2).to_bytes((len(encoded_data1) + 7) // 8, byteorder='big')
                
                # Write the length of the encoded data (in bits)
                f.write(struct.pack('i', len(encoded_data1)))  
                
                # Write the encoded binary data
                f.write(byte_array)
                
                # Save Huffman codes as a serialized dictionary
                huffman_codes_bytes = str(huffman_codes1).encode()
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
    scale = 50.0 / quality_factor
<<<<<<< HEAD
    # Scale the quantization matrix and round values
    scaled_matrix = np.floor((quant_matrix * scale) + 0.5)
    # Clip values to ensure a minimum of 1 (adjusted after normalization)
    normalized_matrix = np.clip(scaled_matrix, 1, None)
=======
    # scale = 50.0 / quality_factor

    # Scale the quantization matrix and round values
    scaled_matrix = np.floor((quant_matrix * scale) + 0.5)

    # Normalize by subtracting 128
    # normalized_matrix = scaled_matrix - 128
    # normalized_matrix = scaled_matrix
#     normalized_matrix[scaled_matrix > 255] = 255
    # normalized_matrix[scaled_matrix <= 0] = 1    

    # Clip values to ensure a minimum of 1 (adjusted after normalization)
    normalized_matrix = np.clip(scaled_matrix, 1, None)

>>>>>>> b1699b4 (Final Commit)
    return normalized_matrix

def is_colored_image(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Check if all channels are identical
        b, g, r = cv2.split(image)
        if (b == g).all() and (g == r).all():
            return 0 
        else:
            return 1
    return 0  
    
def encoder_main(input_image_path, output_file, quality_factor):
    # Load the grayscale image
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError("Image not found or invalid format.")    
    is_colored = is_colored_image(image)
    if is_colored==1:
        print("Image is colored")
        # Convert image to YCbCr for better compression performance
        ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel, cb_channel, cr_channel = cv2.split(ycbcr_image)

        # Downsample chroma channels
        cb_channel_downsampled = cb_channel[::2, ::2]
        cr_channel_downsampled = cr_channel[::2, ::2]

        # Process Y, Cb, Cr channels (store downsampled Cb and Cr)
        channels = [y_channel, cb_channel_downsampled, cr_channel_downsampled]
        # print()
        # print("Y channel shape:", y_channel.shape, "dtype:", y_channel.dtype)
        # print("Cb channel shape:", cb_channel_downsampled.shape, "dtype:", cb_channel_downsampled.dtype)
        # print("Cr channel shape:", cr_channel_downsampled.shape, "dtype:", cr_channel_downsampled.dtype)
        # print()
        padded_channels = []
        encoded_data_per_channel = []
        huffman_codes_per_channel = []
        for channel_idx, channel in enumerate(channels): 
            h, w = channel.shape
            pad_h = (8 - (h % 8)) % 8
            pad_w = (8 - (w % 8)) % 8
            padded_channel = np.pad(channels[channel_idx], ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            h_padded, w_padded = padded_channel.shape
            # print(f"in encoder - {(h_padded, w_padded)}")
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
            scaled_quant_matrix = scale_quantization_matrix(base_quant_matrix, quality_factor)
            # Apply DCT and quantization
            quantized_image = apply_dct_and_quantize(padded_channel, scaled_quant_matrix)
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
            encoded_data_per_channel.append(encoded_data)
            huffman_codes_per_channel.append(huffman_codes)

        # Save to file
        # print("Before Saving")
        # print((h + pad_h, w + pad_w))
        # print(len(encoded_data_per_channel), len(encoded_data_per_channel[0]), len(encoded_data_per_channel[1]), len(encoded_data_per_channel[2]))
        # print(len(huffman_codes_per_channel), len(huffman_codes_per_channel[0]), len(huffman_codes_per_channel[1]), len(huffman_codes_per_channel[2]))
        # print(quality_factor)
        
        save_to_binary_file(output_file, is_colored, (h_padded, w_padded), encoded_data_per_channel, huffman_codes_per_channel, quality_factor)
    else:
        print("Image is grey scale - encoder")
        image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
        h, w = image.shape
        # Calculate padding
        pad_h = (8 - (h % 8)) % 8  # Padding needed for height
        pad_w = (8 - (w % 8)) % 8  # Padding needed for width
        # Pad the image with zeros
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        h_padded, w_padded = padded_image.shape
        
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
<<<<<<< HEAD
        save_to_binary_file(output_file, is_colored, (h_padded, w_padded), encoded_data, huffman_codes, quality_factor)
=======
        save_to_binary_file(output_file, (h_padded, w_padded), encoded_data, huffman_codes, quality_factor,False)
        
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
        print(f'Original size - {(h, w)} ')
        for channel_idx, channel in enumerate(channels):  # Use 'channel_idx' for clarity
            # Pad each channel to ensure dimensions are divisible by 8
            
            pad_h = (8 - (h % 8)) % 8
            pad_w = (8 - (w % 8)) % 8
            
            padded_channel = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            h_padded, w_padded = padded_channel.shape
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
            # quant_matrix = base_quant_matrix if channel_idx == 0 else base_quant_matrix * 2
            scaled_quant_matrix = scale_quantization_matrix(base_quant_matrix, quality_factor)
            # Apply DCT and quantization
            quantized_image = apply_dct_and_quantize(padded_channel, scaled_quant_matrix)

            # Perform RLE and zigzag
            block_size = 8
            h_blocks = h_padded // block_size
            w_blocks = w_padded // block_size
            encoded_blocks = []
            print(f'padded check wala {(h_padded, w_padded)}')
            for i in range(h_blocks):
                for j in range(w_blocks):
                    # block = quantized_image[i:i + block_size, j:j + block_size]
                    block = quantized_image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
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
        print()
        print("Before Saving")
        print((h + pad_h, w + pad_w))
        print(len(encoded_data_per_channel), len(encoded_data_per_channel[0]))
        print(len(huffman_codes_per_channel), len(huffman_codes_per_channel[0]))
        print(quality_factor)
        
        save_to_binary_file(output_file, (h + pad_h, w + pad_w), encoded_data_per_channel, huffman_codes_per_channel, quality_factor,is_colored)
>>>>>>> b1699b4 (Final Commit)
