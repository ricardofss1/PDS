import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


data = np.load("fir_dataset.npz")
specs = data['specs']
coefs = data['coefs']
orders = data['orders']

print(specs.shape)  # (10000, 5)
print(coefs.shape)  # (10000, 256)
print(orders.shape) # (10000,)


# 1) Carregar dataset gerado
data = np.load("fir_dataset.npz")
specs = data["specs"]   # (n_samples, 5) → fc, trans, Rp, As, order
coefs = data["coefs"]   # (n_samples, Nmax)
orders = data["orders"] # (n_samples,)

print("Specs shape:", specs.shape)
print("Coefs shape:", coefs.shape)
print("Orders shape:", orders.shape)

# 2) Mostrar algumas especificações
print("\nExemplo de 5 especificações de filtros:")
for i in range(5):
    fc, trans, Rp, As, order = specs[i]
    print(f"Filtro {i}: fc={fc:.3f}, trans={trans:.3f}, Rp={Rp:.2f} dB, As={As:.1f} dB, ordem={int(order)}")

# 3) Função para plotar coeficientes e resposta em frequência
def plot_filter(i):
    h = coefs[i, :int(orders[i])]  # pega apenas os coeficientes válidos
    fc, trans, Rp, As, order = specs[i]

    # Resposta em frequência
    w, H = signal.freqz(h, worN=1024, fs=2.0)  # fs=2.0 porque normalizamos [0,1]
    H_db = 20 * np.log10(np.abs(H) + 1e-8)

    plt.figure(figsize=(12,4))

    # Coeficientes
    plt.subplot(1,2,1)
    plt.stem(h)
    plt.title(f"Coeficientes do Filtro #{i} (ordem={int(order)})")
    plt.xlabel("n")
    plt.ylabel("h[n]")

    # Resposta em frequência
    plt.subplot(1,2,2)
    plt.plot(w, H_db)
    plt.axvline(fc, color='red', linestyle='--', label="fc")
    plt.title("Resposta em Frequência")
    plt.xlabel("Frequência Normalizada [× Nyquist]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# 4) Visualizar alguns filtros aleatórios
for idx in np.random.choice(len(specs), size=3, replace=False):
    plot_filter(idx)