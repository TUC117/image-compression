import numpy as np
from scipy.fftpack import dct, idct

standard_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def quantize_block(dct_block, quantization_matrix):
    return np.round(dct_block / quantization_matrix)

def dequantize_block(q_block, quantization_matrix):
    return q_block * quantization_matrix


def split_image_into_blocks(image, block_size=8):
    gray_image = image.convert("L")
    image_array = np.array(gray_image, dtype=np.float32)
    h, w = image_array.shape
    h -= h % block_size
    w -= w % block_size
    image_array = image_array[:h, :w]
    blocks = [image_array[i:i+block_size, j:j+block_size] 
                for i in range(0, h, block_size) for j in range(0, w, block_size)]
    return blocks, h, w

def apply_dct_to_blocks(blocks):
    return [dct(dct(block.T, norm='ortho').T, norm='ortho') for block in blocks]
