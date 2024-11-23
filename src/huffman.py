import heapq
from collections import defaultdict

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequency):
    heap = [HuffmanNode(char, freq) for char, freq in frequency.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    return heap[0]

def generate_codes(node, code="", huffman_codes=None):
    if huffman_codes is None:  # Initialize a new dictionary for every call
        huffman_codes = {}
    if node is None:
        return
    if node.char is not None:
        huffman_codes[node.char] = code
    generate_codes(node.left, code + "0", huffman_codes)
    generate_codes(node.right, code + "1", huffman_codes)
    return huffman_codes


def huffman_encode(data):
    frequency = defaultdict(int)
    for char in data:
        frequency[char] += 1
    root = build_huffman_tree(frequency)
    huffman_codes = generate_codes(root)
    encoded_data = ''.join(huffman_codes[char] for char in data)
    return encoded_data, huffman_codes

def huffman_decode(encoded_data, huffman_codes):
    reverse_codes = {v: k for k, v in huffman_codes.items()}
    current_code = ""
    decoded_data = []
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_codes:
            decoded_data.append(reverse_codes[current_code])
            current_code = ""
    return decoded_data
