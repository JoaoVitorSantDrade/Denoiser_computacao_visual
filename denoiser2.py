from skimage import io, color
from skimage.restoration import denoise_tv_bregman
import matplotlib.pyplot as plt

# Load the image
image = io.imread('lena_denoised_1.tif')

# Extract RGB channels if it's a color image
if len(image.shape) == 3 and image.shape[2] == 4:
    image = image[:, :, :3]  # Extract RGB channels, ignore alpha channel

# Convert to grayscale
image_gray = color.rgb2gray(image)

# Denoise using Total Variation Denoising
denoised_image = denoise_tv_bregman(image_gray, weight=100)

# Display the original and denoised images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_gray, cmap='gray')
plt.title('Original Grayscale Image')

plt.subplot(1, 2, 2)
plt.imshow(denoised_image, cmap='gray')
plt.title('Denoised Image')

plt.imsave('lena_denoised_2.tif', denoised_image, cmap='gray', format='tiff')

plt.show()