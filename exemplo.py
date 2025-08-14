import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Carregar o sinal
fs, x = wavfile.read('sinal_1.wav')  # fs = 16 kHz (taxa de amostragem)
x = x / np.max(np.abs(x))  # Normaliza o sinal

# FFT
N = len(x)
X = np.fft.fft(x)
freqs = np.fft.fftfreq(N, d=1/fs)[:N//2]  # Frequências positivas (0 a fs/2)

plt.figure(figsize=(10, 4))
plt.plot(freqs, np.abs(X[:N//2]))
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')
#plt.xlim([500, 1000])
plt.title('Espectro do Sinal (FFT)')
plt.grid(True)
plt.show()