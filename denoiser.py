import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage import io, color

def radius_low_pass(radius, rows, cols):
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols))
    r = np.arange(rows)
    c = np.arange(cols)
    mask = mask * (np.exp(-((r - crow) ** 2 + (c - ccol) ** 2) / (2 * radius ** 2)))
    return mask

def horizontal_low_pass(cutoff_frequency, rows, cols):
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols))
    c = np.arange(cols)
    mask = mask * (np.exp(-((c - ccol) ** 2) / (2 * (cutoff_frequency * cols) ** 2)))
    return mask

# Load the image
image = io.imread('lena_Ruidosa.tif')

# Extract RGB channels if it's a color image
if len(image.shape) == 3 and image.shape[2] == 4:
    image = image[:, :, :3]  # Extract RGB channels, ignore alpha channel


# Convert to grayscale if it's a color image
image = color.rgb2gray(image)

# Apply 2D FFT
f_transformed = fft2(image)

# Shift zero frequency components to the center
f_transformed_shifted = fftshift(f_transformed)

# Visualize the spectrum before filtering
plt.imshow(np.log(1 + np.abs(f_transformed_shifted)), cmap='gray')
plt.title('Frequency Spectrum Before Filtering')
plt.show()

rows, cols = image.shape
# Apply the filter in the frequency domain
f_transformed_shifted_filtered = f_transformed_shifted * horizontal_low_pass(0.1, rows,cols)

# Visualize the spectrum after filtering
plt.imshow(np.log(1 + np.abs(f_transformed_shifted_filtered)), cmap='gray')
plt.title('Frequency Spectrum After Filtering')
plt.show()

# Inverse FFT to get the denoised image
denoised_image = np.abs(ifft2(ifftshift(f_transformed_shifted_filtered)))

# Display the original and denoised images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(denoised_image, cmap='gray')
plt.title('Denoised Image')

plt.imsave('lena_denoised_1.tif', denoised_image, cmap='gray', format='tiff')

plt.show()
