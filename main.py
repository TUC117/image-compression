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
    quality_factor = int(sys.argv[2])
    image_name = input_image.split('/')[-1].split('.')[0]
    encoded_data_dir = os.path.join(os.getcwd(), 'data/', 'output/','encoded_data/', image_name)
    if not os.path.isdir(encoded_data_dir):
        os.mkdir(encoded_data_dir)        
    compressed_images_dir = os.path.join(os.getcwd(), 'data/', 'output/', image_name)
    if not os.path.isdir(compressed_images_dir):
        os.mkdir(compressed_images_dir)               
    if quality_factor == -1:
<<<<<<< HEAD
        for qf in [1, 3, 5, 10, 15, 20, 25, 40, 50, 70, 90]:
=======
        
        for qf in [1,3 ,5, 10, 15, 20, 25,40, 50, 70, 90]:
            # quality_factor = int(input('Enter quality factor: '))
>>>>>>> b1699b4 (Final Commit)
            print(f'For quality factor = {qf}')
            me = str(qf)
            if len(me) == 1:
                me = "0" + me
            encoded_data_path = os.path.join(encoded_data_dir, image_name)
<<<<<<< HEAD
            encoded_data_path += f'-{me}.bin'
            compressed_image_path = os.path.join(compressed_images_dir, image_name)
            compressed_image_path += f'-{me}.png'
=======
            me = str(qf)
            if len(me)==1:
                me = "0"+me
            encoded_data_path += f'-{me}.bin'
            
            compressed_image_path = os.path.join(compressed_images_dir, image_name)
            compressed_image_path += f'-{me}.png'
            
            # print(image_name, encoded_data_path, compressed_image_path)
>>>>>>> b1699b4 (Final Commit)
            encoder_main(input_image_path=input_image, output_file=encoded_data_path, quality_factor=qf)
            decoder_main(input_file=encoded_data_path, output_image_path=compressed_image_path, input_image_path=input_image)
            original_image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
            input_image_size = original_image.shape[0] * original_image.shape[1] 
            output_image_size = os.path.getsize(encoded_data_path) 
        plot_main(original_image_path=input_image, compressed_images_dir=compressed_images_dir)
    else:
        print(f'For quality factor = {quality_factor}')
<<<<<<< HEAD
        me = str(quality_factor)
        if me == 1:
            me = "0" + me            
        encoded_data_path = os.path.join(encoded_data_dir, image_name)
        encoded_data_path += f'-{me}.bin'
        compressed_image_path = os.path.join(compressed_images_dir, image_name)
        compressed_image_path += f'-{me}.png'
=======
        # quality_factor = int(input('Enter quality factor: '))
        print(f'For quality factor = {quality_factor}')
        me = str(quality_factor)
        if len(me)==1:
                me = "0"+me
        encoded_data_path = os.path.join(encoded_data_dir, image_name)
        encoded_data_path += f'-{me}.bin'
        
        compressed_image_path = os.path.join(compressed_images_dir, image_name)
        compressed_image_path += f'-{me}.png'
        
        # print(image_name, encoded_data_path, compressed_image_path)
>>>>>>> b1699b4 (Final Commit)
        encoder_main(input_image_path=input_image, output_file=encoded_data_path, quality_factor=quality_factor)
        decoder_main(input_file=encoded_data_path, output_image_path=compressed_image_path, input_image_path=input_image)
        original_image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
        input_image_size = original_image.shape[0] * original_image.shape[1]
        output_image_size = os.path.getsize(encoded_data_path)
        print(f'Input Image size = {input_image_size}')
        print(f'Compressed Image size = {output_image_size}')
        print(f'Compressed to a factor of = {(1 - (output_image_size / input_image_size))*100}%')