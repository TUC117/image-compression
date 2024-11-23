from src.huffman import huffman_decode,huffman_encode
import struct

rle1=  [(64.0, 1), (1.0, 1), (-0.0, 5), (-1.0, 1), (-999,)]
rle2=  [(60.0, 1), (2.0, 1), (-5.0, 1), (-0.0, 1), (-999,)]
rle3=  [(-3.0, 1), (1.0, 3), (7.0, 1), (10.0, 1), (-999,)]

def save_to_binary_file(output_file, encoded_data, huffman_codes):
    print("COLOR IMAGE _ SAVE TO BINARY")
    with open(output_file, "wb") as f:
        # Save the shape and quality factor
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


def load_from_binary_file(input_file):
    with open(input_file, "rb") as f:
        encoded_data = []
        huffman_codes = []

        # Read encoded data and Huffman codes for each channel
        for _ in range(2):  # Iterate for each channel (R, G, B)
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

    return encoded_data, huffman_codes

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
huffman_codes_list = []
encoded_data_list = []
encoded_data_1, huffman_codes_1 = huffman_encode(rle1)
encoded_data_list.append(encoded_data_1)
huffman_codes_list.append(huffman_codes_1)
print(huffman_codes_1)
encoded_data_2, huffman_codes_2 = huffman_encode(rle2)
encoded_data_list.append(encoded_data_2)
huffman_codes_list.append(huffman_codes_2)
print(huffman_codes_2)
# encoded_data_3, huffman_codes_3 = huffman_encode(rle3)
print("I am groot")



# encoded_data_list.append(encoded_data_3)


# huffman_codes_list.append(huffman_codes_3)

print(rle1)
print(rle2)
# print(rle3)
# print("new")
# print(encoded_data_list)
for a in huffman_codes_list:
    print(a)
    
save_to_binary_file('test.bin', encoded_data_list, huffman_codes_list)

encoded_data, huffman_codes = load_from_binary_file('test.bin')
for encoded_, huffman_code in zip(encoded_data, huffman_codes):
    decodeddata = huffman_decode(encoded_,huffman_code)
    print("hi")
    print(decodeddata)
    print(huffman_code)