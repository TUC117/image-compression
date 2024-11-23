% Step 1: Create 8x8 matrices for Red, Green, and Blue channels
redChannel = uint8(randi([0, 255], 8, 8));   % Random values between 0 and 255
greenChannel = uint8(randi([0, 255], 8, 8));
blueChannel = uint8(randi([0, 255], 8, 8));

% Step 2: Combine channels into an 8x8x3 image
colorImage = cat(3, redChannel, greenChannel, blueChannel);

% Step 3: Display the image (optional)
imshow(colorImage);
title('Generated 8x8 Color Image');

% Step 4: Save the image
imwrite(colorImage, '8x8_color_image.png');
