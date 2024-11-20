import numpy as np
import cv2

def apply_dct_and_quantize(image, quant_matrix, block_size=8):
    """
    Applies 2D DCT and quantization to non-overlapping blocks of the image.
    """
    h, w = image.shape
    dct_quantized = np.zeros_like(image, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            dct_block = cv2.dct(block.astype(np.float32))
            quantized_block = quantize(dct_block, quant_matrix)
            dct_quantized[i:i+block_size, j:j+block_size] = quantized_block
    return dct_quantized


def apply_idct_and_dequantize(quantized_image, quant_matrix, block_size=8):
    """
    Applies dequantization and then 2D IDCT to non-overlapping blocks of the quantized image.
    """
    h, w = quantized_image.shape
    reconstructed_image = np.zeros_like(quantized_image, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            quantized_block = quantized_image[i:i+block_size, j:j+block_size]
            dequantized_block = dequantize(quantized_block, quant_matrix).astype(np.float32)
            idct_block = cv2.idct(dequantized_block)
            reconstructed_image[i:i+block_size, j:j+block_size] = idct_block
    return np.clip(reconstructed_image, 0, 255).astype(np.uint8)

def quantize(block, quant_matrix):
    return np.round(block / quant_matrix)

def dequantize(block, quant_matrix):
    return block * quant_matrix

def run_length_encode(data):
    """
    Perform run-length encoding on a list of data.
    """
    rle = []
    prev_val = data[0]
    count = 1
    for val in data[1:]:
        if val == prev_val:
            count += 1
        else:
            rle.append((prev_val, count))
            prev_val = val
            count = 1
    rle.append((prev_val, count))  # Append the last value
    return rle

def run_length_decode(rle):
    """
    Decode run-length encoded data.
    """
    decoded = []
    for value, count in rle:
        decoded.extend([value] * count)
    return decoded

def zigzag_scan(block):
    """
    Perform zigzag scanning on an 8x8 block.
    """
    h, w = block.shape
    assert h == w, "Block must be square for zigzag scan."
    result = []
    for sum_idx in range(h + w - 1):
        if sum_idx % 2 == 0:
            # Even: traverse diagonally upwards
            for i in range(max(0, sum_idx - w + 1), min(sum_idx + 1, h)):
                result.append(block[i, sum_idx - i])
        else:
            # Odd: traverse diagonally downwards
            for i in range(max(0, sum_idx - h + 1), min(sum_idx + 1, w)):
                result.append(block[sum_idx - i, i])
    return result


def inverse_zigzag_scan(data, block_size=8):
    """
    Reconstruct a block from zigzag-scanned data.
    """
    block = np.zeros((block_size, block_size), dtype=np.float32)
    h, w = block.shape
    idx = 0
    for sum_idx in range(h + w - 1):
        if sum_idx % 2 == 0:
            # Even: traverse diagonally upwards
            for i in range(max(0, sum_idx - w + 1), min(sum_idx + 1, h)):
                block[i, sum_idx - i] = data[idx]
                idx += 1
        else:
            # Odd: traverse diagonally downwards
            for i in range(max(0, sum_idx - h + 1), min(sum_idx + 1, w)):
                block[sum_idx - i, i] = data[idx]
                idx += 1
    return block

def save_to_file(file_path, shape, encoded_data, huffman_codes, quality_factor):
    print(shape)
    with open(file_path, 'w') as f:
        f.write(f"{shape[0]},{shape[1]}\n")
        f.write(f"{quality_factor}\n")
        f.write(f"{encoded_data}\n")
        f.write(f"{huffman_codes}\n")

def load_from_file(file_path):
    with open(file_path, 'r') as f:
        shape = tuple(map(int, f.readline().strip().split(',')))
        quality_factor = int(f.readline().strip())
        encoded_data = f.readline().strip()
        huffman_codes = eval(f.readline().strip())
    print(shape)
    return shape, encoded_data, huffman_codes, quality_factor

def save_to_binary_file(file_path, shape, encoded_data, huffman_codes, quality_factor):
    with open(file_path, 'wb') as f:
        f.write(f"{shape[0]} {shape[1]}\n".encode('utf-8'))
        f.write(f"{quality_factor}\n".encode('utf-8'))
        f.write(encoded_data)  # Write binary data directly
        f.write(f"{str(huffman_codes)}\n".encode('utf-8'))

def load_from_binary_file(file_path):
    with open(file_path, 'rb') as f:
        shape = tuple(map(int, f.readline().decode('utf-8').split()))
        quality_factor = int(f.readline().decode('utf-8').strip())
        encoded_data = f.read(len(encoded_data))  # Adjust as per your encoded data size
        huffman_codes = eval(f.readline().decode('utf-8').strip())
    return shape, encoded_data, huffman_codes, quality_factor

