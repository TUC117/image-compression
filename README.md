# image-compression

image-compression/
├── README.md
├── requirements.txt
├── main.py
├── src/
|   ├── dct.py
|   ├── huffman.py
|   ├── metrics.py
├── data/
   ├── input/
   ├── output/ 

# To-Do List

- [ ] Split image into non-overlapping blocks.
- [ ] Implement 2D DCT on each block.
- [ ] Implement quantization with adjustable quality factors.
- [ ] Implement Huffman coding for compressed data.
- [ ] Write compressed data to a file and create a function to read it.
- [ ] Implement decompression and image reconstruction.
- [ ] Calculate RMSE and BPP metrics.
- [ ] Plot RMSE vs. BPP for different quality factors and multiple images.
