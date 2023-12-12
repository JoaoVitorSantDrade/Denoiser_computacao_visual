import numpy as np
import matplotlib.pyplot as plt

# Leitura da Imagem
read = plt.imread('images/lena 256x256.tif')
f = read.astype(float)

N, _ = f.shape
maximo = np.max(f)

# Cálculo da Transformada de Fourier da Imagem

# Adição de Ruído à Imagem Original
fre = 100 / N
R = np.zeros_like(f)

for I in range(N):
    for J in range(N):
        R[J, I] = 400 * np.sin((2 * np.pi * fre) * I)

fn = f + R

# Plotting
plt.figure()

plt.subplot(2, 2, 1)
plt.imshow(f, cmap='gray', vmax=maximo)
plt.title('Imagem original')

plt.subplot(2, 2, 2)
plt.imshow(fn, cmap='gray', vmax=maximo)
plt.title('Imagem ruidosa')

F = np.fft.fft2(f)
FN = np.fft.fft2(fn)

plt.subplot(2, 2, 3)
c = 15 / np.log10(np.max(np.abs(F)))  # Constante de escala
D = c * np.log(1 + np.abs(F))
z = np.abs(D)
plt.imshow(z)
plt.title('Espectro da DFT de f em Escala Logarítmica')

plt.subplot(2, 2, 4)
cn = 15 / np.log10(np.max(np.abs(FN)))  # Constante de escala
DN = cn * np.log(1 + np.abs(FN))
zn = np.abs(DN)
plt.imshow(zn)
plt.title('Espectro da DFT de fn em Escala Logarítmica')

# Cálculo do SNR
soma1 = np.sum((f - fn)**2)
soma2 = np.sum(f**2)
snr = np.abs(10 * np.log10(soma1 / soma2))

print(f"SNR: {snr}")

# Salvar a imagem ruidosa
plt.imsave('lena_Ruidosa.tif', fn, cmap='gray', format='tiff')

plt.show()