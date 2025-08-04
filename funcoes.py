import numpy as np

def dft_matricial(x):

    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))

    W = np.exp(-2j * np.pi * k * n / N)

    X = np.dot(W, x)

    return X

def zero_padding(sinal, quantidade_zeros=None):
    N_original = len(sinal)

    if quantidade_zeros is None:
        # Expandir até a próxima potência de 2
        N_novo = 2 ** int(np.ceil(np.log2(N_original)))
    else:
        # adiciona a quantidade especificada de zeros
        N_novo = N_original + quantidade_zeros

    # novo array preenchido com zeros
    sinal_padded = np.zeros(N_novo, dtype=sinal.dtype)
    
    # Copia o sinal original para o início com os zeros no final
    sinal_padded[:N_original] = sinal

    return sinal_padded


    
def fft_dec(x):

    N = len(x)

    if N == 1:
        return x

    pares = x[::2]
    impares = x[1::2]

    X_par = fft_dec(pares) # indices pares
    X_impar = fft_dec(impares) # indices ímpares

    k = np.arange(N // 2)
    W_N = np.exp(-2j * np.pi * k / N)

    X = np.zeros(N, dtype=np.complex128)
    X[:N//2] = X_par + W_N * X_impar
    X[N//2:] = X_par - W_N * X_impar

    return X