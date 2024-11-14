from PIL import Image
from src.dct import split_image_into_blocks, apply_dct_to_blocks, quantize_block, dequantize_block
from src.huffman import build_huffman_tree, huffman_codes
from src.metrics import calculate_rmse, calculate_bpp

def compress_image(filename, quality_factor=1.0):
    # Load and process image
    image = Image.open(filename)
    blocks, h, w = split_image_into_blocks(image)

    # Apply DCT
    dct_blocks = apply_dct_to_blocks(blocks)

    # Quantize blocks
    quantized_blocks = [quantize_block(block, standard_quantization_matrix * quality_factor) for block in dct_blocks]

    # Encode blocks with Huffman
    flat_data = [item for block in quantized_blocks for row in block for item in row]
    root = build_huffman_tree(flat_data)
    codes = huffman_codes(root)
    
    # Encode data
    encoded_data = ''.join(codes[value] for value in flat_data)
    
    # Save encoded data, dimensions, etc.
    # Save or display RMSE, BPP as needed

if __name__ == "__main__":
    compress_image("data/input/sample.jpg", quality_factor=1.0)
