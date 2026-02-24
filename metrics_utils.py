# metrics_utils.py
import os, json, csv
import torch
import torch.nn as nn
import numpy as np

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def save_checkpoint(model, path, extra=None):
    ensure_dir(os.path.dirname(path))
    payload = {"state_dict": model.state_dict()}
    if extra: payload.update(extra)
    torch.save(payload, path)

def append_csv(path, fieldnames, row):
    ensure_dir(os.path.dirname(path))
    new = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new: w.writeheader()
        w.writerow(row)

def write_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def read_json(path, default=None):
    if not os.path.isfile(path): return default
    try:
        with open(path, "r") as f: return json.load(f)
    except Exception:
        return default

def spectral_norm_stats(model: nn.Module):
    sv_list = []
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            W = m.weight.detach().float().cpu()
            W2d = W.view(W.shape[0], -1)
            # Power iteration for top singular value
            if W2d.numel() == 0: continue
            u = torch.randn(W2d.size(0))
            for _ in range(8):
                v = torch.mv(W2d.t(), u); v = v / (v.norm() + 1e-12)
                u = torch.mv(W2d, v); u = u / (u.norm() + 1e-12)
            sigma = torch.dot(u, torch.mv(W2d, v)).item()
            sv_list.append(sigma)
    if not sv_list:
        return {"spectral_norm_mean": float("nan"), "spectral_norm_max": float("nan")}
    return {
        "spectral_norm_mean": float(np.mean(sv_list)),
        "spectral_norm_max": float(np.max(sv_list)),
    }

def gram_linear(X):
    return X @ X.t()

def cka_linear(X, Y):
    # X,Y: [N, D] torch tensors
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    Kx = gram_linear(X)
    Ky = gram_linear(Y)
    hsic = (Kx * Ky).sum()
    denom = (torch.linalg.norm(Kx) * torch.linalg.norm(Ky)) + 1e-12
    return (hsic / denom).item()

def collect_features_sequential_penultimate(model: nn.Sequential, dataloader, device="cpu", max_batches=None):
    model.eval()
    feats = []
    with torch.no_grad():
        for bi, (xb, _) in enumerate(dataloader):
            xb = xb.to(device)
            z = xb
            penultimate = None
            for i, layer in enumerate(model):
                z = layer(z)
                if i == len(model)-2:
                    penultimate = z
            feats.append(penultimate.detach().cpu())
            if max_batches is not None and (bi+1) >= max_batches:
                break
    return torch.cat(feats, dim=0)

def eval_accuracy_proxy(model, data_loader, device="cpu"):
    # For control deltas regression, we compute 1 - MSE as a proxy "accuracy"
    model.eval()
    correct, total = 0.0, 0
    import torch
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            mse = torch.mean((pred - yb)**2).item()
            proxy = max(0.0, 1.0 - mse)
            correct += proxy * xb.size(0)
            total += xb.size(0)
    return correct / max(1, total)

def pruning_robustness_auc(model_ctor, base_state, val_loader, device="cpu", sparsities=(0.0, 0.2, 0.4, 0.6, 0.8)):
    from torch.nn.utils import prune
    import copy
    accs = []
    for s in sparsities:
        m = model_ctor().to(device)
        sd = base_state["state_dict"] if isinstance(base_state, dict) and "state_dict" in base_state else base_state
        m.load_state_dict(sd, strict=True)
        params_to_prune = []
        for mod in m.modules():
            if isinstance(mod, nn.Linear):
                params_to_prune.append((mod, 'weight'))
        if s > 0 and params_to_prune:
            for (mod, pname) in params_to_prune:
                prune.l1_unstructured(mod, name=pname, amount=s)
        acc = eval_accuracy_proxy(m, val_loader, device=device)
        accs.append(acc)
    # trapezoid AUC
    xs = list(sparsities); auc = 0.0
    for i in range(1, len(xs)):
        dx = xs[i]-xs[i-1]
        auc += 0.5 * dx * (accs[i] + accs[i-1])
    return {"sparsities": xs, "accuracies": accs, "pruning_auc": auc}
