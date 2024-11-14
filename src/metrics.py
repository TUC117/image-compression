import numpy as np

def calculate_rmse(original, reconstructed):
    return np.sqrt(np.mean((original - reconstructed) ** 2))

def calculate_bpp(encoded_data, h, w):
    return len(encoded_data) / (h * w)
