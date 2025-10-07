import time, torch
from my_tqdm import Pbar
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# ---------- tiny helpers ----------
def log2(x):
    return torch.log(x) / torch.log(torch.tensor(2.0, device=x.device, dtype=x.dtype))


def poly_eval(coeffs, x):
    y = torch.zeros_like(x, dtype=coeffs.dtype, device=x.device)
    D = coeffs.numel() - 1
    for pwr in range(D, -1, -1):
        y = y * x + coeffs[D - pwr]
    return y


# ---------- metrics: logits vs probs ----------
def js_divergence_logits_vs_probs(pred_logits, target_probs, dim=-1, eps=1e-12):
    P = F.softmax(pred_logits, dim=dim).clamp_min(eps)
    Q = target_probs.clamp_min(eps)
    Q = Q / Q.sum(dim=dim, keepdim=True).clamp_min(eps)
    M = 0.5 * (P + Q)
    kl_pm = torch.sum(P * (log2(P + eps) - log2(M + eps)), dim=dim)
    kl_qm = torch.sum(Q * (log2(Q + eps) - log2(M + eps)), dim=dim)
    return 0.5 * (kl_pm + kl_qm)


@torch.no_grad()
def kl_target_pred_probs_vs_logits(target_probs, pred_logits, dim=-1, eps=1e-12):
    # KL(target || pred)
    Q = target_probs.clamp_min(eps)
    Q = Q / Q.sum(dim=dim, keepdim=True).clamp_min(eps)
    P = F.softmax(pred_logits, dim=dim).clamp_min(eps)
    return torch.sum(Q * (log2(Q + eps) - log2(P + eps)), dim=dim)


@torch.no_grad()
def kl_pred_target_logits_vs_probs(pred_logits, target_probs, dim=-1, eps=1e-12):
    # KL(pred || target)
    P = F.softmax(pred_logits, dim=dim).clamp_min(eps)
    Q = target_probs.clamp_min(eps)
    Q = Q / Q.sum(dim=dim, keepdim=True).clamp_min(eps)
    return torch.sum(P * (log2(P + eps) - log2(Q + eps)), dim=dim)


@torch.no_grad()
def evaluate_all_metrics(model, pki_idx, y_probs, batch_size=8192):
    """
    pki_idx: [N,3] with columns (p,k,i_in)
    y_probs: [N,A]
    """
    device = next(model.parameters()).device
    n = 0
    js_sum = kl_qp_sum = kl_pq_sum = 0.0
    for s in range(0, pki_idx.size(0), batch_size):
        pki_b = pki_idx[s : s + batch_size].to(device)
        y_b = y_probs[s : s + batch_size].to(device)
        logits = model(pki_b)
        js = js_divergence_logits_vs_probs(logits, y_b, dim=-1)
        klq = kl_target_pred_probs_vs_logits(y_b, logits, dim=-1)
        klp = kl_pred_target_logits_vs_probs(logits, y_b, dim=-1)
        js_sum += js.sum().item()
        kl_qp_sum += klq.sum().item()
        kl_pq_sum += klp.sum().item()
        n += js.numel()
    return (
        js_sum / max(n, 1),
        kl_qp_sum / max(n, 1),  # KL(target||pred)
        kl_pq_sum / max(n, 1),  # KL(pred||target)
    )


# ---------- model ----------
class MinimalModel(nn.Module):
    """
    Now takes (p, k, i_in) as inputs; targets are distributions over answer positions a=0..A-1.

    logits(a | p,k,i_in) =
          w_pos * exp( - (a - p)^2 / (2 σ_p(p)^2) )
        + gate_k[k] * 1{a = k}
        + gate_i[i_in] * 1{a = i_in}

    σ_p(p) = (quadratic in normalized p) + ε   (you kept this unconstrained/ReLU-free in your snippet)
    gate_k[k], gate_i[i] ∈ ℝ (unconstrained learned tables)
    """

    def __init__(self, P: int, K: int, I: int, A: int):
        super().__init__()
        self.P, self.K, self.I, self.A = P, K, I, A

        self.w_pos = nn.Parameter(torch.tensor(1.0))

        # σ_p(p): quadratic in normalized p (here you used [0,1] scaling)
        sig_init = torch.zeros(3)
        sig_init[-1] = 3.0
        self.sigmap_coeffs = nn.Parameter(sig_init)

        # key gate table (unconstrained)
        self.gate_k = nn.Parameter(torch.zeros(K))

        # new: i gate table (unconstrained)
        self.gate_i = nn.Parameter(torch.zeros(I))

    def _pn(self, p1d):  # your snippet used [0,1] normalization
        return p1d.float() / self.P

    def _sigmap(self, p1d):
        pn = self._pn(p1d)
        raw = poly_eval(self.sigmap_coeffs.view(-1), pn)
        return raw.view(-1, 1) + 1e-6
        # If you want positivity:
        # return self.softplus(raw).view(-1,1) + 1e-6

    def forward(self, pki_idx):
        # pki_idx: [B,3] with (p,k,i_in)
        p = pki_idx[:, 0].view(-1, 1)
        k = pki_idx[:, 1].view(-1, 1)
        i_in = pki_idx[:, 2].view(-1, 1)
        a = torch.arange(self.A, device=pki_idx.device).view(1, -1)

        # positional Gaussian over answer a centered at p with width σ_p(p)
        sp = self._sigmap(p.view(-1))  # [B,1]
        Kpos = torch.exp(-((a - p) ** 2) / (2.0 * (sp**2)))  # [B,A]

        # key spike and i spike
        Kkey = (a == k).float()  # [B,A]
        Ki = (a == i_in).float()  # [B,A]

        gate_k = self.gate_k.index_select(0, k.view(-1)).view(-1, 1)  # [B,1]
        gate_i = self.gate_i.index_select(0, i_in.view(-1)).view(-1, 1)  # [B,1]

        logits = self.w_pos * Kpos + gate_k * Kkey + gate_i * Ki
        return logits


# ---------- training (JS) ----------
def train_minimal_model_3in(
    fv_probs: torch.Tensor,  # [P,K,I,A] probabilities
    epochs: int = 5000,
    batch_size: int = 8192,
    patience: int = 500,
    lr_wpos: float = 5e-2,
    lr_tables_shapes: float = 5e-2,  # σ_p coeffs + gate_k + gate_i
    seed: int = 0,
    device: str | None = None,
    plot: bool = True,
):
    """
    Trains MinimalModel against fv_probs shaped [P,K,I,A].
    Reports variance and 95% confidence intervals for evaluation metrics.
    """
    import scipy.stats

    def mean_var_ci(arr):
        arr = np.asarray(arr)
        mean = arr.mean()
        var = arr.var(ddof=1)
        n = arr.shape[0]
        if n > 1:
            se = arr.std(ddof=1) / np.sqrt(n)
            ci95 = scipy.stats.t.interval(0.95, n - 1, loc=mean, scale=se)
        else:
            ci95 = (mean, mean)
        return mean, var, ci95

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    P, K, I, A = fv_probs.shape
    model = MinimalModel(P, K, I, A).to(device)

    # build (p,k,i) grid and split
    pp, kk, ii = torch.meshgrid(torch.arange(P), torch.arange(K), torch.arange(I), indexing="ij")
    pki = torch.stack([pp, kk, ii], dim=-1).reshape(-1, 3)  # [N,3]
    y_probs = fv_probs.reshape(P * K * I, A)  # [N,A]

    N = pki.size(0)
    perm = torch.randperm(N)
    n_train = int(0.70 * N)
    n_val = int(0.15 * N)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    train_loader = DataLoader(TensorDataset(pki[train_idx], y_probs[train_idx]), batch_size=batch_size, shuffle=True)

    # param groups
    weights = [model.w_pos]  # alias to w_p_gauss
    tables_shapes = [model.sigmap_coeffs, model.gate_k, model.gate_i]

    opt = torch.optim.Adam(
        [
            {"params": weights, "lr": lr_wpos},
            {"params": tables_shapes, "lr": lr_tables_shapes},
        ],
        betas=(0.9, 0.999),
    )

    hist = {"train_js": [], "val_js": []}
    best_val = float("inf")
    best_state = None
    bad = 0
    t0 = time.time()

    pbar = Pbar(total=epochs)

    for ep in range(1, epochs + 1):
        # train
        model.train()
        tr_js_sum = tr_n = 0
        for pki_b, yb in train_loader:
            pki_b, yb = pki_b.to(device), yb.to(device)
            logits = model(pki_b)
            js = js_divergence_logits_vs_probs(logits, yb, dim=-1)
            loss = js.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_js_sum += js.sum().item()
            tr_n += js.numel()

        # val
        model.eval()
        val_js, val_kl_qp, val_kl_pq = evaluate_all_metrics(model, pki[val_idx], y_probs[val_idx])

        hist["train_js"].append(tr_js_sum / max(tr_n, 1))
        hist["val_js"].append(val_js)

        if val_js < best_val - 1e-6:
            best_val = val_js
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

        pbar.set_description(
            f"Train JS: {100-(hist['train_js'][-1]*100):.2f} | Val JS: {100-(val_js*100):.2f} | KL(t||p): {val_kl_qp:.4f} | KL(p||t): {val_kl_pq:.4f} | Loss: {loss:.4f}"
        )
        pbar.update(1)

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate metrics and also collect per-sample values for variance/CI
    def eval_metrics_and_collect(model, pki_idx, y_probs):
        model.eval()
        with torch.no_grad():
            logits = model(pki_idx.to(device))
            y_probs = y_probs.to(device)
            js = js_divergence_logits_vs_probs(logits, y_probs, dim=-1).cpu().numpy()
            # KL(t||p) and KL(p||t)
            p_pred = torch.softmax(logits, dim=-1)
            kl_qp = (y_probs * (torch.log(y_probs + 1e-12) - torch.log(p_pred + 1e-12))).sum(dim=-1).cpu().numpy()
            kl_pq = (p_pred * (torch.log(p_pred + 1e-12) - torch.log(y_probs + 1e-12))).sum(dim=-1).cpu().numpy()
        return js, kl_qp, kl_pq

    val_js, val_kl_qp, val_kl_pq = evaluate_all_metrics(model, pki[val_idx], y_probs[val_idx])
    test_js, test_kl_qp, test_kl_pq = evaluate_all_metrics(model, pki[test_idx], y_probs[test_idx])
    runtime_s = time.time() - t0

    # Collect per-sample metrics for variance/CI
    val_js_arr, val_kl_qp_arr, val_kl_pq_arr = eval_metrics_and_collect(model, pki[val_idx], y_probs[val_idx])
    test_js_arr, test_kl_qp_arr, test_kl_pq_arr = eval_metrics_and_collect(model, pki[test_idx], y_probs[test_idx])

    # Compute mean, variance, and 95% CI for each metric
    import numpy as np

    val_js_mean, val_js_var, val_js_ci = mean_var_ci(val_js_arr)
    val_kl_qp_mean, val_kl_qp_var, val_kl_qp_ci = mean_var_ci(val_kl_qp_arr)
    val_kl_pq_mean, val_kl_pq_var, val_kl_pq_ci = mean_var_ci(val_kl_pq_arr)
    test_js_mean, test_js_var, test_js_ci = mean_var_ci(test_js_arr)
    test_kl_qp_mean, test_kl_qp_var, test_kl_qp_ci = mean_var_ci(test_kl_qp_arr)
    test_kl_pq_mean, test_kl_pq_var, test_kl_pq_ci = mean_var_ci(test_kl_pq_arr)

    if plot:
        plt.figure(figsize=(6.2, 4.2))
        plt.plot(hist["train_js"], label="train JS")
        plt.plot(hist["val_js"], label="val JS")
        plt.xlabel("epoch")
        plt.ylabel("JS")
        plt.title(f"JS (best val={val_js:.4f})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # visualize learned tables and σ_p(p)
        with torch.no_grad():
            gate_k = model.gate_k.detach().cpu().numpy()
            gate_i = model.gate_i.detach().cpu().numpy()
            p = torch.arange(P)
            pn = p.float() / P  # matches _pn()
            sig = (poly_eval(model.sigmap_coeffs.detach().cpu(), pn) + 1e-6).numpy()

        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
        ax[0].plot(gate_k, marker="o")
        ax[0].set_title("gate_k[k]")
        ax[0].set_xlabel("k")
        ax[0].grid(alpha=0.3)
        ax[1].plot(gate_i, marker="o")
        ax[1].set_title("gate_i[i]")
        ax[1].set_xlabel("i")
        ax[1].grid(alpha=0.3)
        ax[2].plot(sig)
        ax[2].set_title("σ_p(p) (raw poly + ε)")
        ax[2].set_xlabel("p")
        ax[2].grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    print(
        f"Early stop at epoch {ep}. "
        f"Val JS: {val_js:.6f} | Test JS: {test_js:.6f} | "
        f"Val KL(t||p): {val_kl_qp:.6f} | Test KL(t||p): {test_kl_qp:.6f} | "
        f"Val KL(p||t): {val_kl_pq:.6f} | Test KL(p||t): {test_kl_pq:.6f} | "
        f"time {runtime_s:.2f}s"
    )
    print("\n--- Variance and 95% CI for evaluation metrics ---")
    print(f"Val JS: mean={val_js_mean:.6f}, var={val_js_var:.6e}, 95% CI=({val_js_ci[0]:.6f}, {val_js_ci[1]:.6f})")
    print(f"Test JS: mean={test_js_mean:.6f}, var={test_js_var:.6e}, 95% CI=({test_js_ci[0]:.6f}, {test_js_ci[1]:.6f})")
    print(
        f"Val KL(t||p): mean={val_kl_qp_mean:.6f}, var={val_kl_qp_var:.6e}, 95% CI=({val_kl_qp_ci[0]:.6f}, {val_kl_qp_ci[1]:.6f})"
    )
    print(
        f"Test KL(t||p): mean={test_kl_qp_mean:.6f}, var={test_kl_qp_var:.6e}, 95% CI=({test_kl_qp_ci[0]:.6f}, {test_kl_qp_ci[1]:.6f})"
    )
    print(
        f"Val KL(p||t): mean={val_kl_pq_mean:.6f}, var={val_kl_pq_var:.6e}, 95% CI=({val_kl_pq_ci[0]:.6f}, {val_kl_pq_ci[1]:.6f})"
    )
    print(
        f"Test KL(p||t): mean={test_kl_pq_mean:.6f}, var={test_kl_pq_var:.6e}, 95% CI=({test_kl_pq_ci[0]:.6f}, {test_kl_pq_ci[1]:.6f})"
    )

    info = {
        "val_js": float(val_js),
        "test_js": float(test_js),
        "val_kl_target_pred": float(val_kl_qp),
        "test_kl_target_pred": float(test_kl_qp),
        "val_kl_pred_target": float(val_kl_pq),
        "test_kl_pred_target": float(test_kl_pq),
        "epochs": ep,
        "time_s": runtime_s,
        "lr_wpos": lr_wpos,
        "lr_tables_shapes": lr_tables_shapes,
        # Add variance and CI info for downstream use
        "val_js_var": float(val_js_var),
        "val_js_ci": tuple(float(x) for x in val_js_ci),
        "test_js_var": float(test_js_var),
        "test_js_ci": tuple(float(x) for x in test_js_ci),
        "val_kl_target_pred_var": float(val_kl_qp_var),
        "val_kl_target_pred_ci": tuple(float(x) for x in val_kl_qp_ci),
        "test_kl_target_pred_var": float(test_kl_qp_var),
        "test_kl_target_pred_ci": tuple(float(x) for x in test_kl_qp_ci),
        "val_kl_pred_target_var": float(val_kl_pq_var),
        "val_kl_pred_target_ci": tuple(float(x) for x in val_kl_pq_ci),
        "test_kl_pred_target_var": float(test_kl_pq_var),
        "test_kl_pred_target_ci": tuple(float(x) for x in test_kl_pq_ci),
    }
    return model, info
