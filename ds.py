import numpy as np
from scipy import signal
from tqdm import tqdm
import os

def sample_spec():
    # retorna um dicionário com especificações válidas (normalizadas, fs=1)
    fcut = np.random.uniform(0.02, 0.45)          # cutoff
    trans = 10**np.random.uniform(np.log10(0.005), np.log10(0.12))  # log-uniform
    Rp = np.random.uniform(0.01, 1.0)             # ripple dB
    As = 10**np.random.uniform(np.log10(30), np.log10(100))  # stopband dB
    order = np.random.randint(8, 129)             # ordem FIR
    # garantir validade
    if fcut + trans >= 0.5:
        trans = 0.5 - fcut - 1e-4
    return {'type':'lowpass','fc':fcut,'trans':trans,'Rp':Rp,'As':As,'order':order}

def design_fir(spec, method='remez'):
    N = spec['order']
    fc = spec['fc']
    trans = spec['trans']
    bands = [0, fc, fc+trans, 0.5]
    desired = [1, 0]
    try:
        if method == 'remez':
            h = signal.remez(N+1, bands, desired, fs=1.0)
        else:
            h = signal.firwin(N+1, fc, window='hamming', fs=1.0)  # <<< corrigido
    except Exception as e:
        # fallback
        h = signal.firwin(N+1, fc, window='hamming', fs=1.0)
    return h

def generate_dataset(n_samples=10000, Nmax=256, out_fname='fir_dataset.npz'):
    specs = []
    coefs = np.zeros((n_samples, Nmax), dtype=np.float32)
    orders = np.zeros(n_samples, dtype=np.int32)

    for i in tqdm(range(n_samples)):
        s = sample_spec()
        h = design_fir(s, method='remez')
        L = len(h)
        if L > Nmax:
            # truncar (ou escolher ordem menor)
            h = h[:Nmax]
            L = Nmax
        coefs[i, :L] = h
        specs.append([s['fc'], s['trans'], s['Rp'], s['As'], s['order']])
        orders[i] = s['order']

    specs = np.array(specs, dtype=np.float32)  # shape (n_samples, n_features)
    np.savez_compressed(out_fname, specs=specs, coefs=coefs, orders=orders)
    print("Saved:", out_fname)

if __name__ == '__main__':
    generate_dataset(n_samples=10000, Nmax=256, out_fname='fir_dataset.npz')
