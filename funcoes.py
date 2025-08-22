import numpy as np
from scipy.fft import ifft

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
    
    # sinal original para o início com os zeros no final
    sinal_padded[:N_original] = sinal

    return sinal_padded


def fft_dec(x):
    """
    Transformada Rápida de Fourier (FFT) recursiva usando o algoritmo Cooley-Tukey.
    Parâmetros:
    x : array_like
        Sinal de entrada no domínio do tempo (real ou complexo)
    
    Retorna:
    X : ndarray
        Espectro de frequência complexo do sinal de entrada
    """
    # comprimento do sinal
    N = len(x)
    
    # Caso base
    # A DFT de um único ponto é o próprio ponto
    if N == 1:
        return x

    # Divide o sinal em componentes pares e ímpares
    pares = x[::2]    # elementos com índices pares: x[0], x[2], x[4], ...
    impares = x[1::2] # elementos com índices ímpares: x[1], x[3], x[5], ...

    # Chamadas recursivas para calcular a FFT
    X_par = fft_dec(pares)   # FFT dos índices pares
    X_impar = fft_dec(impares) # FFT dos índices ímpares

    # Pré-calcula os fatores de rotação
    k = np.arange(N // 2)  # Índices de 0 a N/2-1
    W_N = np.exp(-2j * np.pi * k / N)  # Fatores de rotação para FFT

    # array de resultado com zeros complexos
    X = np.zeros(N, dtype=np.complex128)
    
    # Combina os resultados
    # Primeira metade: X[k] = X_par[k] + W_N^k * X_impar[k]
    X[:N//2] = X_par + W_N * X_impar
    
    # Segunda metade: X[k + N/2] = X_par[k] - W_N^k * X_impar[k]
    X[N//2:] = X_par - W_N * X_impar

    return X


def _ifft_rec(y):
    """
    Função recursiva auxiliar para cálculo da IFFT (sem normalização).
    
    Implementa o algoritmo inverso de Cooley-Tukey para calcular
    a Transformada de Fourier Inversa de forma recursiva.
    
    Parâmetros:
    y : array_like
        Espectro de frequência complexo (resultado da FFT)
    
    Retorna:
    Y : ndarray
        Sinal reconstruído no domínio do tempo (sem normalização)
    """
    # comprimento do espectro de frequência
    N = len(y)

    if N == 1:
        return y

    # Divide o espectro em componentes pares e ímpares
    # Mesma estratégia de divisão da FFT, mas aplicada ao domínio da frequência
    pares = y[::2]    # Componentes de frequência com índices pares
    impares = y[1::2] # Componentes de frequência com índices ímpares

    # Chamadas recursivas
    Y_par = _ifft_rec(pares)   # IFFT dos componentes pares
    Y_impar = _ifft_rec(impares) # IFFT dos componentes ímpares

    # Pré-calcula os fatores de rotação inversos
    # Para IFFT usamos o conjugado complexo: W_N^{-k} = e^(+2πj*k/N)
    k = np.arange(N // 2)  # Índices de 0 a N/2-1
    W_N = np.exp(2j * np.pi * k / N)  # Fatores de rotação para IFFT (sinal positivo)

    Y = np.zeros(N, dtype=np.complex128)
    
    # Combina os resultados usando a fórmula de reconstrução
    Y[:N//2] = Y_par + W_N * Y_impar
    
    Y[N//2:] = Y_par - W_N * Y_impar

    return Y


def ifft_dec(y):
    """
    Transformada Rápida de Fourier Inversa (IFFT) recursiva.
    
    Calcula a transformada inversa normalizada do espectro de frequência
    para recuperar o sinal original no domínio do tempo.
    
    Parâmetros:
    y : array_like
        Espectro de frequência complexo (resultado da FFT)
    
    Retorna:
    y_reconstructed : ndarray
        Sinal original reconstruído no domínio do tempo
    """
    # A IFFT é calculada aplicando o algoritmo inverso e depois
    # normalizando pelo comprimento do sinal (1/N)
    # A normalização é necessária porque a definição matemática da IFFT inclui
    # o fator 1/N, enquanto a FFT não tem essa normalização
    return _ifft_rec(y) / len(y)


def convolucao_circular(x, h):
    """
    Calcula a convolução circular entre dois sinais x e h.
    
    Parâmetros:
    -----------
    x : array_like
        Primeiro sinal de entrada
    h : array_like
        Segundo sinal de entrada
    
    Retorna:
    --------
    y : ndarray
        Resultado da convolução circular de tamanho N
    """
    N = max(len(x), len(h))
    
    # Aplica zero-padding
    x_padded = zero_padding(x, (N - len(x)))
    h_padded = zero_padding(h, (N - len(h)))

    x_padded = zero_padding(x)
    h_padded = zero_padding(h)

    # Calcula as DFTs
    X = fft_dec(x_padded)
    H = fft_dec(h_padded)
    
    # Multiplica no domínio da frequência
    Y = X * H
    
    # Calcula a IDFT e retorna a parte real
    y = ifft(Y).real
    
    return y

def convolucao_circular_direta(x, h):
    """
    Implementação direta da convolução circular
    """
    N = max(len(x), len(h))
    
    x_padded = np.pad(x, (0, N - len(x)))
    h_padded = np.pad(h, (0, N - len(h)))
    
    y = np.zeros(N)
    for n in range(N):
        for k in range(N):
            y[n] += x_padded[k] * h_padded[(n - k) % N]
    
    return y


def overlap_add(x, h, M0=None):
    x = np.asarray(x)
    h = np.asarray(h)

    L = len(h)               
    # Define o tamanho do bloco se não especificado
    if M0 is None:
        M0 = 4 * L  # Valor padrão 4 vezes o comprimento de h   

    N = M0 + L - 1  
    # filtro com zero-padding até próxima potência de 2
    h_padded = zero_padding(h, N - L)
    h_padded = zero_padding(h_padded)  
    Nfft = len(h_padded)

    # FFT do filtro
    H = fft_dec(h_padded)

    # número de blocos
    num_blocos = int(np.ceil(len(x) / M0))

    # saída
    y = np.zeros(len(x) + L - 1)

    for i in range(num_blocos):
        # extrair bloco
        inicio = i * M0
        fim = min(inicio + M0, len(x))
        bloco = x[inicio:fim]

        # Zero-padding do bloco
        bloco_padded = zero_padding(bloco, Nfft - len(bloco))

        # FFT do bloco
        X = fft_dec(bloco_padded)
        Y = X * H

        # IFFT
        y_bloco = np.fft.ifft(Y).real
        # comprimento válido do resultado
        len_valida = len(bloco) + L - 1
        
        # Acumula no resultado
        y[inicio:inicio+len_valida] += y_bloco[:len_valida]

    return y