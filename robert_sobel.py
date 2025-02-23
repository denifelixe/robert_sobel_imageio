# Langkah 1: Instalasi dan Import Library
# !pip install imageio matplotlib numpy scipy

import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d #Import convolve2d from scipy

# Langkah 2: Membaca Gambar
gambar = imageio.imread('2.png')
gambar_gray = np.dot(gambar[..., :3], [0.2989, 0.5870, 0.1140])  # Konversi ke grayscale

# Langkah 3: Visualisasi Gambar Asli
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gambar_gray, cmap='gray')
plt.title('Gambar Asli (Grayscale)')
plt.axis('off')

# Langkah 4: Implementasi Operator Robert
def robert_operator(image):
    gx = np.array([[1, 0], [0, -1]])
    gy = np.array([[0, 1], [-1, 0]])

    #Note that for scipy.signal.convolve2d the kernel should be reversed
    Ix = convolve2d(image, gx[::-1, ::-1], mode='same')
    Iy = convolve2d(image, gy[::-1, ::-1], mode='same')

    magnitude = np.sqrt(Ix**2 + Iy**2)
    magnitude = np.uint8(magnitude)

    return magnitude

robert_edges = robert_operator(gambar_gray)

# Langkah 5: Implementasi Operator Sobel
def sobel_operator(image):
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    #Note that for scipy.signal.convolve2d the kernel should be reversed
    Ix = convolve2d(image, gx[::-1, ::-1], mode='same')
    Iy = convolve2d(image, gy[::-1, ::-1], mode='same')

    magnitude = np.sqrt(Ix**2 + Iy**2)
    magnitude = np.uint8(magnitude)

    return magnitude

sobel_edges = sobel_operator(gambar_gray)

# Langkah 6: Visualisasi Hasil Deteksi Tepi
plt.subplot(1, 2, 2)
plt.imshow(robert_edges, cmap='gray')
plt.title('Deteksi Tepi dengan Operator Robert')
plt.axis('off')

plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(sobel_edges, cmap='gray')
plt.title('Deteksi Tepi dengan Operator Sobel')
plt.axis('off')
plt.show()

# Langkah 7: Analisis
print("Analisis:")
print("Operator Robert menggunakan dua masker 2x2 yang saling tegak lurus untuk mendeteksi tepi secara diagonal.")
print("Operator Sobel menggunakan dua masker 3x3 yang lebih besar untuk mendeteksi tepi secara horizontal dan vertikal.")
print("Dari hasil visualisasi, kita dapat melihat bahwa:")
print("- Operator Robert lebih sensitif terhadap tepi diagonal dan mungkin menghasilkan tepi yang lebih tipis.")
print("- Operator Sobel menghasilkan tepi yang lebih halus dan lebih baik dalam mendeteksi tepi horizontal dan vertikal.")
print("Pilihan operator yang tepat tergantung pada jenis tepi yang ingin dideteksi dan kebutuhan aplikasi.")
