import heapq
from collections import defaultdict

class HuffmanNode:
    def __init__(self, value=None, freq=0, left=None, right=None):
        self.value = value
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    freq_dict = defaultdict(int)
    for item in data:
        freq_dict[item] += 1
    heap = [HuffmanNode(value=k, freq=v) for k, v in freq_dict.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(freq=node1.freq + node2.freq, left=node1, right=node2)
        heapq.heappush(heap, merged)
    return heap[0]  # root node

def huffman_codes(root):
    codes = {}
    def _generate_codes(node, code=""):
        if node is not None:
            if node.value is not None:
                codes[node.value] = code
            _generate_codes(node.left, code + "0")
            _generate_codes(node.right, code + "1")
    _generate_codes(root)
    return codes
