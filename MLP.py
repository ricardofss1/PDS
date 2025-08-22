"""
train_filter_net.py


Requisitos:
- Python 3.8+
- numpy, scipy (só para geração/visualização, não estritamente necessário),
- torch
- matplotlib (para plots opcionais)

Instalação (ex. pip):
pip install numpy scipy torch matplotlib

Uso:
python train_filter_net.py
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt

# -----------------------
# Dataset wrapper
# -----------------------
class FIRDataset(Dataset):
    """
    Espera um arquivo .npz com chaves:
      specs: shape (N, n_features)
      coefs: shape (N, Nmax)  # coeficientes zero-padded
      orders: shape (N,)      # ordem real (N = ordem+1 coeficientes válidos)
    """
    def __init__(self, npz_path, specs_scaler=None, coefs_scaler=None):
        data = np.load(npz_path)
        self.specs = data["specs"].astype(np.float32)
        self.coefs = data["coefs"].astype(np.float32)
        self.orders = data["orders"].astype(np.int32)

        # basic sanity
        assert self.specs.shape[0] == self.coefs.shape[0] == self.orders.shape[0]

        # scalers: tuples (mean, std) for standardization OR None
        self.specs_scaler = specs_scaler
        self.coefs_scaler = coefs_scaler

        # if scalers not provided, compute from data (caller can override later with train stats)
        if self.specs_scaler is None:
            mean = self.specs.mean(axis=0)
            std = self.specs.std(axis=0)
            std[std == 0] = 1.0
            self.specs_scaler = (mean.astype(np.float32), std.astype(np.float32))

        if self.coefs_scaler is None:
            mean = self.coefs.mean(axis=0)
            std = self.coefs.std(axis=0)
            std[std == 0] = 1.0
            self.coefs_scaler = (mean.astype(np.float32), std.astype(np.float32))

        # Precompute mask indicies (boolean mask per sample) for efficient retrieval
        # valid_len = orders + 1 (since order = N-1). But earlier code stores order as exact number of taps? 
        # We'll assume orders array stores "order" (N), i.e., number of taps (consistent with previous code where 'order' was number of taps).
        # If orders actually mean polynomial order (N-1), adjust by +1 or -1 accordingly.
        # Here we handle both: if any orders == 0, assume they meant order==num_taps and >=1.
        # We'll interpret orders as num_taps (L). If your orders represent N, change accordingly.
        self.Nmax = self.coefs.shape[1]
        self.valid_lens = self.orders.copy()
        # safety: clamp
        self.valid_lens = np.clip(self.valid_lens, 1, self.Nmax).astype(np.int32)

    def __len__(self):
        return self.specs.shape[0]

    def __getitem__(self, idx):
        spec = self.specs[idx]
        coefs = self.coefs[idx]
        L = int(self.valid_lens[idx])
        mask = np.zeros(self.Nmax, dtype=np.float32)
        mask[:L] = 1.0  # valid coefficients are the first L entries

        # apply scaling (standardization)
        spec_mean, spec_std = self.specs_scaler
        coefs_mean, coefs_std = self.coefs_scaler

        spec_scaled = (spec - spec_mean) / spec_std
        coefs_scaled = (coefs - coefs_mean) / coefs_std

        return {
            "spec": torch.from_numpy(spec_scaled),
            "coefs": torch.from_numpy(coefs_scaled),
            "mask": torch.from_numpy(mask),
            "valid_len": L,
            "raw_spec": torch.from_numpy(spec.astype(np.float32))
        }

    def get_scalers(self):
        return self.specs_scaler, self.coefs_scaler

# -----------------------
# Small MLP model (baseline)
# -----------------------
class MLPFilterNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(512,512,256), dropout=0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -----------------------
# Frequency response helper (torch)
# -----------------------
def batch_freq_response(coefs_batch, n_fft=512):
    """
    coefs_batch: tensor shape (B, Nmax) real
    returns: mag (B, n_fft//2 + 1) linear magnitude
    Uses rFFT to compute frequency response at n_fft points (0..Nyquist)
    """
    # pad or truncate coefs to n_fft length for FFT
    B, Nmax = coefs_batch.shape
    # zero-pad to n_fft
    if Nmax < n_fft:
        pad = torch.zeros((B, n_fft - Nmax), device=coefs_batch.device, dtype=coefs_batch.dtype)
        x = torch.cat([coefs_batch, pad], dim=1)
    else:
        x = coefs_batch[:, :n_fft]
    # rfft -> shape (B, n_fft//2 + 1), complex
    X = torch.fft.rfft(x, n=n_fft, dim=1)
    mag = torch.abs(X)  # linear magnitude
    return mag

# -----------------------
# Train / Eval loops
# -----------------------
def train_epoch(model, loader, optimizer, coefs_scaler, device, alpha=1.0, beta=0.5, n_fft=512):
    model.train()
    total_loss = 0.0
    total_coef_loss = 0.0
    total_freq_loss = 0.0
    criterion = nn.MSELoss(reduction='sum')  # will divide by N manually

    coefs_mean, coefs_std = coefs_scaler
    coefs_mean = torch.from_numpy(coefs_mean).to(device)
    coefs_std = torch.from_numpy(coefs_std).to(device)

    for batch in loader:
        spec = batch["spec"].to(device)            # (B, nf)
        coefs = batch["coefs"].to(device)          # (B, Nmax) scaled
        mask = batch["mask"].to(device)            # (B, Nmax)
        B = spec.shape[0]

        pred = model(spec)                         # (B, Nmax) scaled prediction

        # coef loss on masked region (unscale first to compute in natural units? it's fine in scaled space)
        coef_loss = criterion(pred * mask, coefs * mask) / mask.sum().clamp(min=1.0)  # avg per valid coef in batch

        # unscale to natural units for freq calc (so FFT uses real-scale magnitudes)
        pred_un = (pred * coefs_std) + coefs_mean
        target_un = (coefs * coefs_std) + coefs_mean

        # Frequency-domain loss: compare magnitude responses
        pred_mag = batch_freq_response(pred_un, n_fft=n_fft)  # (B, n_fft//2+1)
        target_mag = batch_freq_response(target_un, n_fft=n_fft)

        # MSE on magnitude (linear)
        freq_loss = criterion(pred_mag, target_mag) / (pred_mag.numel() / B)

        loss = alpha * coef_loss + beta * freq_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        total_coef_loss += coef_loss.item() * B
        total_freq_loss += freq_loss.item() * B

    N = len(loader.dataset)
    return total_loss / N, total_coef_loss / N, total_freq_loss / N

def eval_epoch(model, loader, coefs_scaler, device, alpha=1.0, beta=0.5, n_fft=512):
    model.eval()
    total_loss = 0.0
    total_coef_loss = 0.0
    total_freq_loss = 0.0
    criterion = nn.MSELoss(reduction='sum')

    coefs_mean, coefs_std = coefs_scaler
    coefs_mean = torch.from_numpy(coefs_mean).to(device)
    coefs_std = torch.from_numpy(coefs_std).to(device)

    with torch.no_grad():
        for batch in loader:
            spec = batch["spec"].to(device)
            coefs = batch["coefs"].to(device)
            mask = batch["mask"].to(device)
            B = spec.shape[0]

            pred = model(spec)

            coef_loss = criterion(pred * mask, coefs * mask) / mask.sum().clamp(min=1.0)

            pred_un = (pred * coefs_std) + coefs_mean
            target_un = (coefs * coefs_std) + coefs_mean

            pred_mag = batch_freq_response(pred_un, n_fft=n_fft)
            target_mag = batch_freq_response(target_un, n_fft=n_fft)
            freq_loss = criterion(pred_mag, target_mag) / (pred_mag.numel() / B)

            loss = alpha * coef_loss + beta * freq_loss

            total_loss += loss.item() * B
            total_coef_loss += coef_loss.item() * B
            total_freq_loss += freq_loss.item() * B

    N = len(loader.dataset)
    return total_loss / N, total_coef_loss / N, total_freq_loss / N

# -----------------------
# Utilities: save / load checkpoint
# -----------------------
def save_checkpoint(path, model, optimizer, epoch, best_val):
    state = {
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val
    }
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt

# -----------------------
# Simple plotting helper
# -----------------------
def plot_prediction(model, dataset, device, idx=None, n_fft=1024):
    model.eval()
    if idx is None:
        idx = np.random.randint(len(dataset))
    sample = dataset[idx]
    spec_raw = sample["raw_spec"].numpy()
    coefs_true = sample["coefs"].numpy()
    mask = sample["mask"].numpy()
    valid_len = sample["valid_len"]
    if isinstance(valid_len, torch.Tensor):
        valid_len = int(valid_len.item())
    else:
        valid_len = int(valid_len)

    # get scalers to unscale
    specs_scaler, coefs_scaler = dataset.get_scalers()
    coefs_mean, coefs_std = coefs_scaler

    # prepare tensors
    spec_scaled = sample["spec"].unsqueeze(0).to(device)
    with torch.no_grad():
        pred_scaled = model(spec_scaled).cpu().numpy()[0]

    pred_un = (pred_scaled * coefs_std) + coefs_mean
    true_un = (coefs_true * coefs_std) + coefs_mean

    h_true = true_un[:valid_len]
    h_pred = pred_un[:valid_len]  # using same effective length

    # freq responses
    from scipy import signal
    w_true, H_true = signal.freqz(h_true, worN=n_fft, fs=2.0)
    w_pred, H_pred = signal.freqz(h_pred, worN=n_fft, fs=2.0)

    H_true_db = 20 * np.log10(np.abs(H_true) + 1e-8)
    H_pred_db = 20 * np.log10(np.abs(H_pred) + 1e-8)

    plt.figure(figsize=(12,5))

    plt.subplot(1,3,1)
    plt.stem(h_true)
    plt.title("Coeficientes verdadeiros (h_true)")
    plt.xlabel("n")

    plt.subplot(1,3,2)
    plt.stem(h_pred)
    plt.title("Coeficientes previstos (h_pred)")
    plt.xlabel("n")

    plt.subplot(1,3,3)
    plt.plot(w_true, H_true_db, label="True")
    plt.plot(w_pred, H_pred_db, label="Pred", linestyle='--')
    plt.title("Resposta em frequência (dB)")
    plt.xlabel("Freq normalizada")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------
# Main training routine
# -----------------------
def run_training(npz_path,
                 out_dir="checkpoints",
                 batch_size=64,
                 epochs=100,
                 lr=1e-3,
                 val_split=0.15,
                 alpha=1.0,
                 beta=0.5,
                 n_fft=512,
                 device=None,
                 hidden_dims=(512,512,256),
                 dropout=0.0,
                 seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load dataset, then split into train/val/test
    full_dataset = FIRDataset(npz_path)
    N = len(full_dataset)
    n_val = int(N * val_split)
    n_test = n_val
    n_train = N - n_val - n_test
    assert n_train > 0, "Dataset muito pequeno"

    train_set, val_set, test_set = random_split(full_dataset, [n_train, n_val, n_test],
                                                generator=torch.Generator().manual_seed(seed))

    # Important: compute scalers from training set and override dataset scalers for all splits
    # We'll compute mean/std across train samples (stack)
    specs_train = np.stack([full_dataset.specs[i] for i in train_set.indices], axis=0)
    coefs_train = np.stack([full_dataset.coefs[i] for i in train_set.indices], axis=0)

    spec_mean = specs_train.mean(axis=0).astype(np.float32)
    spec_std = specs_train.std(axis=0).astype(np.float32); spec_std[spec_std==0]=1.0
    coefs_mean = coefs_train.mean(axis=0).astype(np.float32)
    coefs_std = coefs_train.std(axis=0).astype(np.float32); coefs_std[coefs_std==0]=1.0

    # re-create dataset objects with scalers
    train_ds = FIRDataset(npz_path, specs_scaler=(spec_mean, spec_std), coefs_scaler=(coefs_mean, coefs_std))
    val_ds = FIRDataset(npz_path, specs_scaler=(spec_mean, spec_std), coefs_scaler=(coefs_mean, coefs_std))
    test_ds = FIRDataset(npz_path, specs_scaler=(spec_mean, spec_std), coefs_scaler=(coefs_mean, coefs_std))

    # but keep same splits as before (random_split produced subsets of indices). We'll wrap them via Subset
    from torch.utils.data import Subset
    train_ds = Subset(train_ds, train_set.indices)
    val_ds = Subset(val_ds, val_set.indices)
    test_ds = Subset(test_ds, test_set.indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    in_dim = spec_mean.shape[0]
    out_dim = coefs_mean.shape[0]

    model = MLPFilterNet(in_dim=in_dim, out_dim=out_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    os.makedirs(out_dir, exist_ok=True)
    best_val = float("inf")
    best_epoch = -1
    patience = 15
    stale = 0

    print("Start training: N_train={}, N_val={}, N_test={}".format(n_train, n_val, n_test))
    for epoch in range(1, epochs+1):
        t0 = time.time()
        train_loss, train_coef_loss, train_freq_loss = train_epoch(model, train_loader, optimizer,
                                                                    (coefs_mean, coefs_std), device,
                                                                    alpha=alpha, beta=beta, n_fft=n_fft)
        val_loss, val_coef_loss, val_freq_loss = eval_epoch(model, val_loader, (coefs_mean, coefs_std), device,
                                                            alpha=alpha, beta=beta, n_fft=n_fft)
        dt = time.time() - t0

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} (coef={train_coef_loss:.6f}, freq={train_freq_loss:.6f}) "
              f"| val_loss={val_loss:.6f} (coef={val_coef_loss:.6f}, freq={val_freq_loss:.6f}) | dt={dt:.1f}s")

        # checkpointing best val
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            ckpt_path = os.path.join(out_dir, "best_model.pth")
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_val)
            print(f"  -> saved best model (val_loss improved)")
            stale = 0
        else:
            stale += 1

        # early stopping
        if stale >= patience:
            print("Early stopping (no improvement in {:.0f} epochs)".format(patience))
            break

    # load best and evaluate on test set
    best_ckpt = os.path.join(out_dir, "best_model.pth")
    if os.path.exists(best_ckpt):
        load_checkpoint(best_ckpt, model, optimizer=None)
        test_loss, test_coef_loss, test_freq_loss = eval_epoch(model, test_loader, (coefs_mean, coefs_std), device,
                                                               alpha=alpha, beta=beta, n_fft=n_fft)
        print(f"Test loss = {test_loss:.6f} (coef={test_coef_loss:.6f}, freq={test_freq_loss:.6f})")
    else:
        print("No checkpoint saved.")

    # final: plot a sample prediction
    try:
        # get dataset object used for plotting (train_ds is a Subset, underlying dataset at dataset.dataset)
        underlying = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
        plot_prediction(model, underlying, device)
    except Exception as e:
        print("Plotting failed:", e)

    return model, (spec_mean, spec_std), (coefs_mean, coefs_std)

# -----------------------
# If executed as script
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="fir_dataset.npz", help="Caminho para o arquivo .npz")
    parser.add_argument("--out", type=str, default="checkpoints", help="Diretório p/ checkpoints")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=1.0, help="peso para loss em coeficientes")
    parser.add_argument("--beta", type=float, default=0.5, help="peso para loss em frequência")
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_training(npz_path=args.data,
                 out_dir=args.out,
                 batch_size=args.batch,
                 epochs=args.epochs,
                 lr=args.lr,
                 val_split=args.val_split,
                 alpha=args.alpha,
                 beta=args.beta,
                 n_fft=args.n_fft,
                 seed=args.seed)
