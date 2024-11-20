import os
import time
import sys
import cv2
import numpy as np
from src.decoder import decoder_main
from src.encoder import encoder_main
from src.genimages import plot_main

if __name__ == "__main__":
    input_image = sys.argv[1]
    # input_dir = sys.argv[1]
    # input_images = os.listdir(input_dir)
    
    quality_factor = int(sys.argv[2])
    # print(input_images)
    # for image in input_images:
        # input_image = os.path.join(input_dir, image)
        # input_image = input("Enter path of the input image: ")
    if quality_factor == -1:
        image_name = input_image.split('/')[-1].split('.')[0]
        
        encoded_data_dir = os.path.join(os.getcwd(), 'data/', 'output/','encoded_data/', image_name)
        if not os.path.isdir(encoded_data_dir):
            os.mkdir(encoded_data_dir)        
            
        compressed_images_dir = os.path.join(os.getcwd(), 'data/', 'output/', image_name)
        if not os.path.isdir(compressed_images_dir):
            os.mkdir(compressed_images_dir)       
            
        
        for qf in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            # quality_factor = int(input('Enter quality factor: '))
            print(f'For quality factor = {qf}')

            encoded_data_path = os.path.join(encoded_data_dir, image_name)
            encoded_data_path += f'-{qf}.bin'
            
            compressed_image_path = os.path.join(compressed_images_dir, image_name)
            compressed_image_path += f'-{qf}.png'
            
            # print(image_name, encoded_data_path, compressed_image_path)
            encoder_main(input_image_path=input_image, output_file=encoded_data_path, quality_factor=qf)
            time.sleep(2)
            
            # print(f'Done Encoding and data stored in {encoded_data_path}')
            decoder_main(input_file=encoded_data_path, output_image_path=compressed_image_path, input_image_path=input_image)
            time.sleep(2)
            
            # print(f'Done Compression and image stored in {compressed_image_path}')
            original_image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
            compressed_image = cv2.imread(compressed_image_path, cv2.IMREAD_GRAYSCALE)
            
            input_image_size = original_image.shape[0] * original_image.shape[1] 
            output_image_size = os.path.getsize(encoded_data_path) 
            
            print(f'Input Image size = {input_image_size}')
            print(f'Compressed Image size = {output_image_size}')
            print(f'Compressed to a factor of = {(1 - (output_image_size / input_image_size))*100}%')
            # time.sleep(2)
            
        plot_main(original_image_path=input_image, compressed_images_dir=compressed_images_dir)
    else:
        print(f'For quality factor = {quality_factor}')
        # quality_factor = int(input('Enter quality factor: '))
        image_name = input_image.split('/')[-1].split('.')[0]
        encoded_data_path = os.path.join(os.getcwd(), 'data/', 'output/','encoded_data/', image_name)
        encoded_data_path += f'-{quality_factor}.bin'
        compressed_image_path = os.path.join(os.getcwd(), 'data/', 'output/', image_name)
        compressed_image_path += f'-{quality_factor}.png'
        # print(image_name, encoded_data_path, compressed_image_path)
        encoder_main(input_image_path=input_image, output_file=encoded_data_path, quality_factor=quality_factor)
        # print(f'Done Encoding and data stored in {encoded_data_path}')
        decoder_main(input_file=encoded_data_path, output_image_path=compressed_image_path, input_image_path=input_image)
        # print(f'Done Compression and image stored in {compressed_image_path}')
        original_image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
        compressed_image = cv2.imread(compressed_image_path, cv2.IMREAD_GRAYSCALE)
        input_image_size = original_image.shape[0] * original_image.shape[1]
        output_image_size = os.path.getsize(encoded_data_path)

        print(f'Input Image size = {input_image_size}')
        print(f'Compressed Image size = {output_image_size}')
        print(f'Compressed to a factor of = {(1 - (output_image_size / input_image_size))*100}%')
    