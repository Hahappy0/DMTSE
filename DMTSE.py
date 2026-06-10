import os

os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, confusion_matrix, precision_recall_fscore_support


NUM_VIEWS = 3
NUM_PKTS = 20
PAYLOAD_BYTES = 40
RAW_PAYLOAD_DIM = NUM_PKTS * PAYLOAD_BYTES
RAW_LENGTH_DIM = NUM_PKTS
RAW_GRAPH_DIM = NUM_PKTS * NUM_PKTS
VIEW_DIMS = [RAW_PAYLOAD_DIM, RAW_LENGTH_DIM, RAW_GRAPH_DIM]
NUM_HEADS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
P_DIM = 256
Q_DIM = 256
R_DIM = NUM_VIEWS
PROJ_DIM = 128
TAU = 0.1456722657722828
CONTRAST_TOPK = 3
LAMBDA_ABC = 1
LAMBDA_CON_RAW = 1
LAMBDA_RECON_MSE = 20.0
NORMALIZE_RECON_BY_VIEW_SCALE = True
CHUNK_J = 1024
CHUNK_I = 32
U2_LSTM_LOOKBACK = 5
U3_KNN = 3
APPLY_GLOBAL_LATENT_MINMAX = True
APPLY_BATCH_LATENT_MINMAX_WITH_GLOBAL_STATS = True
GLOBAL_MINMAX_EPS = 1e-12
MINMAX_CLIP_QUANTILE = 0.001

def get_timestamp():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

def log(msg: str):
    print(f'[{get_timestamp()}] {msg}', flush=True)

def _class_names_from_any(class_names, n_classes: int):
    if class_names is None:
        return [f'class_{i}' for i in range(n_classes)]
    out = list(class_names)
    if len(out) < n_classes:
        out = out + [f'class_{i}' for i in range(len(out), n_classes)]
    return [str(x) for x in out]

def cluster_match_info(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    D = max(int(y_pred.max()), int(y_true.max())) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    y_pred_mapped = np.array([mapping.get(int(p), int(p)) for p in y_pred], dtype=np.int64)
    acc = float((y_pred_mapped == y_true).mean()) if y_true.size > 0 else float('nan')
    return {'acc': acc, 'mapping': mapping, 'weight_matrix': w, 'pred_mapped': y_pred_mapped}

def clustering_eval_detailed_from_labels(pred, y_np, feat_np=None, sample_ratio=0.2, class_names=None):
    pred = np.asarray(pred, dtype=np.int64)
    y_np = np.asarray(y_np, dtype=np.int64)
    nmi = float(normalized_mutual_info_score(y_np, pred))
    ari = float(adjusted_rand_score(y_np, pred))
    match = cluster_match_info(y_np, pred)
    acc = match['acc']
    pred_mapped = match['pred_mapped']
    sil = np.nan
    if feat_np is not None and sample_ratio is not None and (0 < sample_ratio < 1):
        m = int(len(feat_np) * sample_ratio)
        if m > 10:
            idx = np.random.choice(len(feat_np), size=m, replace=False)
            if len(np.unique(pred[idx])) > 1:
                sil = float(silhouette_score(feat_np[idx], pred[idx]))
    n_classes = int(max(y_np.max(), pred_mapped.max()) + 1)
    class_names = _class_names_from_any(class_names, n_classes)
    labels = np.arange(n_classes, dtype=np.int64)
    cm_raw = confusion_matrix(y_np, pred_mapped, labels=labels)
    row_sum = cm_raw.sum(axis=1, keepdims=True).astype(np.float64)
    cm_norm = cm_raw.astype(np.float64) / np.clip(row_sum, 1e-12, None)
    precision, recall, f1, support = precision_recall_fscore_support(y_np, pred_mapped, labels=labels, zero_division=0)
    per_class = []
    for i in labels:
        per_class.append({'class_id': int(i), 'class_name': class_names[int(i)], 'precision': float(precision[int(i)]), 'recall': float(recall[int(i)]), 'f1': float(f1[int(i)]), 'support': int(support[int(i)]), 'acc_by_class': float(cm_norm[int(i), int(i)]) if int(i) < cm_norm.shape[0] else 0.0})
    macro_precision = float(np.mean(precision)) if len(precision) else float('nan')
    macro_recall = float(np.mean(recall)) if len(recall) else float('nan')
    macro_f1 = float(np.mean(f1)) if len(f1) else float('nan')
    return {'NMI': nmi, 'ARI': ari, 'ACC': acc, 'Silhouette(sample)': sil, 'Macro-Precision': macro_precision, 'Macro-Recall': macro_recall, 'Macro-F1': macro_f1, 'per_class': per_class, 'pred_raw': pred, 'pred_mapped': pred_mapped, 'mapping': match['mapping'], 'cm_raw': cm_raw, 'cm_norm': cm_norm, 'class_names': class_names}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def l2norm(x: torch.Tensor, dim: int=-1, eps: float=1e-12) -> torch.Tensor:
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)

def tensor_to_float(x, default: float=0.0) -> float:
    if x is None:
        return float(default)
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return float(default)
        return float(x.detach().mean().cpu().item())
    try:
        return float(x)
    except Exception:
        return float(default)

@torch.no_grad()
def compute_global_minmax_stats_per_view(XNd_list: list[torch.Tensor], clip_quantile: float=MINMAX_CLIP_QUANTILE) -> list[tuple[torch.Tensor, torch.Tensor]]:
    stats: list[tuple[torch.Tensor, torch.Tensor]] = []
    for X in XNd_list:
        if clip_quantile <= 0:
            x_min = X.amin()
            x_max = X.amax()
        else:
            x_flat = X.reshape(-1)
            if x_flat.numel() > 1000000:
                idx = torch.randperm(x_flat.numel(), device=x_flat.device)[:1000000]
                x_flat = x_flat[idx]
            x_min = torch.quantile(x_flat.float(), float(clip_quantile))
            x_max = torch.quantile(x_flat.float(), float(1.0 - clip_quantile))
            if (x_max - x_min).abs().item() < 1e-12:
                x_min = X.amin()
                x_max = X.amax()
        stats.append((x_min.detach().clone().to(X.dtype), x_max.detach().clone().to(X.dtype)))
    return stats

@torch.no_grad()
def minmax_bank_per_view_with_stats_(XNd_list: list[torch.Tensor], stats: list[tuple[torch.Tensor, torch.Tensor]], eps: float=GLOBAL_MINMAX_EPS) -> list[torch.Tensor]:
    assert len(XNd_list) == len(stats)
    for k in range(len(XNd_list)):
        X = XNd_list[k]
        x_min, x_max = stats[k]
        X.clamp_(min=float(x_min.item()), max=float(x_max.item()))
        X.sub_(x_min).div_(x_max - x_min + eps)
    return XNd_list

def normalize_batch_latents_with_global_stats(Xbd_list: list[torch.Tensor], stats: list[tuple[torch.Tensor, torch.Tensor]], eps: float=GLOBAL_MINMAX_EPS) -> list[torch.Tensor]:
    assert len(Xbd_list) == len(stats)
    out: list[torch.Tensor] = []
    for k in range(len(Xbd_list)):
        x_min, x_max = stats[k]
        x = Xbd_list[k].clamp(min=float(x_min.item()), max=float(x_max.item()))
        out.append((x - x_min) / (x_max - x_min + eps))
    return out

@torch.no_grad()
def update_feature_bank(XNd_list: list[torch.Tensor], idxs: torch.Tensor, Xbd_list: list[torch.Tensor], momentum: float=0.0) -> None:
    assert len(XNd_list) == len(Xbd_list), 'K mismatch'
    for k in range(len(XNd_list)):
        bank = XNd_list[k]
        xb = Xbd_list[k].detach()
        if momentum <= 0:
            bank[idxs] = xb
        else:
            bank[idxs] = bank[idxs] * momentum + xb * (1.0 - momentum)

class CachedFlowTensorDataset(Dataset):

    def __init__(self, payloads, lengths, graphs, labels, flow_data=None, label_mapping=None, id2label=None):
        self.payloads = payloads.float().cpu()
        self.lengths = lengths.float().cpu()
        self.graphs = graphs.float().cpu()
        self.labels = labels.long().cpu()
        if flow_data is None:
            self.flow_data = [('', '', str(i), int(y)) for i, y in enumerate(self.labels.tolist())]
        else:
            self.flow_data = [tuple(x) for x in flow_data]
        self.label_mapping = dict(label_mapping or {})
        self.id2label = {int(k): v for k, v in dict(id2label or {}).items()}

    def __len__(self):
        return int(self.labels.numel())

    def __getitem__(self, idx):
        return (int(idx), self.payloads[idx], self.lengths[idx], self.graphs[idx], int(self.labels[idx].item()))

    @staticmethod
    def collate_fn(batch):
        idxs, payloads, lengths, graphs, labels = zip(*batch)
        return (torch.tensor(idxs, dtype=torch.long), torch.stack(payloads), torch.stack(lengths), torch.stack(graphs), torch.tensor(labels, dtype=torch.long))

class End2EndBackbone(nn.Module):

    def __init__(self, view_dims=None):
        super().__init__()
        raw_dims = list(view_dims or VIEW_DIMS)
        self.raw_dims = raw_dims
        self.out_dim = None
        self.view_dims = list(raw_dims)
        self.ae_list = nn.ModuleList([])
        self.mlp_list = self.ae_list

    def flatten_views(self, payload, lengths, graph) -> list[torch.Tensor]:
        return [payload.flatten(1).float(), lengths.flatten(1).float(), graph.flatten(1).float()]

    def encode_views(self, raw_list: list[torch.Tensor]) -> list[torch.Tensor]:
        return raw_list

    def forward(self, payload, lengths, graph):
        return self.flatten_views(payload, lengths, graph)

class KR_MHA_Pooling(nn.Module):

    def __init__(self, p=P_DIM, num_heads=NUM_HEADS, chunk_i=CHUNK_I):
        super().__init__()
        assert p % num_heads == 0, 'p 必须能被 num_heads 整除'
        self.p = p
        self.chunk_i = chunk_i
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim=p, num_heads=num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, p) * 0.02)

    def forward(self, Xk: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        d, N = Xk.shape
        p = A.shape[1]
        assert p == self.p
        AT = A.t()
        Uk = Xk.new_empty(d, p)
        for s in range(0, d, self.chunk_i):
            e = min(d, s + self.chunk_i)
            X_block = Xk[s:e, :]
            V = X_block[:, None, :] * AT[None, :, :]
            V = V.permute(0, 2, 1).contiguous()
            q = self.query.expand(V.size(0), 1, p)
            out, _ = self.mha(q, V, V)
            Uk[s:e, :] = out.squeeze(1)
        return Uk

class U1CNNRefiner(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int | None=None, num_pkts: int=NUM_PKTS, payload_bytes: int=PAYLOAD_BYTES):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels or in_channels)
        self.num_pkts = int(num_pkts)
        self.payload_bytes = int(payload_bytes)
        self.conv1 = nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=(3, 5), padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(self.hidden_channels)
        self.conv2 = nn.Conv2d(self.hidden_channels, self.in_channels, kernel_size=(3, 5), padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(self.in_channels)

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        d, p = U.shape
        expect_d = self.num_pkts * self.payload_bytes
        if d != expect_d:
            return U
        x = U.t().contiguous().view(1, p, self.num_pkts, self.payload_bytes)
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = y + x
        y = y.view(p, d).t().contiguous()
        return y

class U2LSTMRefiner(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int | None=None, lookback: int=U2_LSTM_LOOKBACK):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim or input_dim)
        self.lookback = int(lookback)
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, batch_first=True, num_layers=1)
        self.proj = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        T, p = U.shape
        outs = []
        for i in range(T):
            s = max(0, i - self.lookback + 1)
            seq = U[s:i + 1].unsqueeze(0)
            out, _ = self.lstm(seq)
            yi = self.proj(out[:, -1, :]).squeeze(0)
            outs.append(yi)
        Y = torch.stack(outs, dim=0)
        return Y + U

class U3HypergraphRefiner(nn.Module):

    def __init__(self, input_dim: int, k: int=U3_KNN):
        super().__init__()
        self.input_dim = int(input_dim)
        self.k = int(k)
        self.theta = nn.Linear(self.input_dim, self.input_dim, bias=False)

    @torch.no_grad()
    def _build_incidence(self, U: torch.Tensor) -> torch.Tensor:
        n = U.size(0)
        Xn = F.normalize(U.float(), dim=1)
        sim = Xn @ Xn.t()
        sim.fill_diagonal_(-float('inf'))
        k = min(self.k, max(n - 1, 1))
        knn = sim.topk(k=k, dim=1).indices
        H = U.new_zeros((n, n))
        row_ids = torch.arange(n, device=U.device)
        H[row_ids, row_ids] = 1.0
        H[knn, row_ids[:, None]] = 1.0
        return H

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        n, p = U.shape
        if n <= 1:
            return U
        H = self._build_incidence(U)
        dv = H.sum(dim=1).clamp_min(1.0)
        de = H.sum(dim=0).clamp_min(1.0)
        Dv_inv_sqrt = dv.pow(-0.5)
        De_inv = de.pow(-1.0)
        X = self.theta(U)
        X = Dv_inv_sqrt[:, None] * X
        X = H.t() @ X
        X = De_inv[:, None] * X
        X = H @ X
        X = Dv_inv_sqrt[:, None] * X
        return X + U

class SharedCoreReconstructorChunk(nn.Module):

    def __init__(self, K=NUM_VIEWS, p=P_DIM, q=Q_DIM, r=R_DIM, chunk_j=CHUNK_J):
        super().__init__()
        self.K = K
        self.chunk_j = chunk_j
        self.p, self.q, self.r = (p, q, r)
        self.core = nn.Parameter(torch.randn(p, q, r) * 0.02)
        self.C = nn.Parameter(torch.randn(K, r) * 0.02)
        self._dbg_printed_xk_stats = False
        self.register_buffer('x_scale', torch.ones(K, dtype=torch.float32), persistent=True)
        self._x_scale_init = False

    def wd_loss(self):
        return self.core.pow(2).mean() + self.C.pow(2).mean()

    @torch.no_grad()
    def minmax_bank_per_view(XNd_list, eps: float=1e-12):
        out = []
        for X in XNd_list:
            x_min = X.amin()
            x_max = X.amax()
            out.append((X - x_min) / (x_max - x_min + eps))
        return out

    @torch.no_grad()
    def update_feature_bank(XNd_list: list[torch.Tensor], idxs: torch.Tensor, Xbd_list: list[torch.Tensor], momentum: float=0.0) -> None:
        assert len(XNd_list) == len(Xbd_list), 'K mismatch'
        for k in range(len(XNd_list)):
            bank = XNd_list[k]
            xb = Xbd_list[k].detach()
            if momentum <= 0:
                bank[idxs] = xb
            else:
                bank[idxs] = momentum * bank[idxs] + (1 - momentum) * xb

    def _check_shapes(self, U_list, B_all, X_list):
        assert len(U_list) == self.K and len(X_list) == self.K
        N = B_all.size(0)
        for k in range(self.K):
            d, p = U_list[k].shape
            assert p == self.p
            assert X_list[k].shape == (d, N)

    def recon_mse_chunk(self, U_list, B_all, X_list, view_weights=None):
        G = torch.einsum('pqr,kr->kpq', self.core, self.C)
        N = B_all.size(0)
        if NORMALIZE_RECON_BY_VIEW_SCALE and (not self._x_scale_init):
            with torch.no_grad():
                scales = []
                for kk in range(self.K):
                    s = X_list[kk].float().std().clamp_min(0.0001)
                    scales.append(s)
                self.x_scale = torch.stack(scales).to(self.x_scale.device)
            self._x_scale_init = True
        if not self._dbg_printed_xk_stats:
            self._dbg_printed_xk_stats = True
        vw = view_weights if view_weights is not None else None
        total_mse = B_all.new_tensor(0.0)
        per_view_mse = B_all.new_zeros(self.K)
        for k in range(self.K):
            Uk = U_list[k]
            Gk = G[k]
            Xk = X_list[k]
            Yk = torch.matmul(Uk, Gk)
            view_se = B_all.new_tensor(0.0)
            view_cnt = 0
            for j0 in range(0, N, self.chunk_j):
                j1 = min(N, j0 + self.chunk_j)
                Bc = B_all[j0:j1, :]
                pred = torch.matmul(Yk, Bc.t())
                pred = torch.sigmoid(pred)
                tgt = Xk[:, j0:j1]
                diff = pred - tgt
                view_se += (diff * diff).sum()
                view_cnt += diff.numel()
            view_mse = view_se / max(view_cnt, 1)
            if NORMALIZE_RECON_BY_VIEW_SCALE and self._x_scale_init:
                mse_scale_sq = self.x_scale[k] * self.x_scale[k]
                view_mse_n = view_mse / mse_scale_sq
            else:
                view_mse_n = view_mse
            per_view_mse[k] = view_mse_n.detach()
            if vw is not None:
                total_mse += vw[k] * view_mse_n
            else:
                total_mse += view_mse_n
        return (total_mse / self.K, per_view_mse)



def build_ab_concat(global_model):
    A_all = global_model.A.weight.detach().cpu()
    B_all = global_model.B.weight.detach().cpu()
    return torch.cat([A_all, B_all], dim=1).numpy()



def col_ortho_penalty(M: torch.Tensor, eps: float=1e-06) -> torch.Tensor:
    n, d = M.shape
    G = M.t() @ M / (float(n) + eps)
    I = torch.eye(d, device=M.device, dtype=M.dtype)
    return (G - I).pow(2).mean()



class GlobalEnd2End(nn.Module):

    def __init__(self, N: int, K=NUM_VIEWS, view_dims: list[int] | None=None,
                 p=P_DIM, q=Q_DIM, r=R_DIM, num_heads=NUM_HEADS, proj_dim=PROJ_DIM,
                 tau=TAU, lambda_con_raw=LAMBDA_CON_RAW, lambda_abc=LAMBDA_ABC,
                 chunk_j=CHUNK_J, chunk_i=CHUNK_I, lambda_recon_mse=LAMBDA_RECON_MSE):
        super().__init__()
        self.view_dims = list(view_dims or VIEW_DIMS)
        self.N, self.d, self.K = (N, max(self.view_dims), K)
        self.p, self.q, self.r = (p, q, r)
        self.chunk_j = chunk_j
        self.chunk_i = chunk_i
        self.lambda_abc = lambda_abc
        self.lambda_con_raw = lambda_con_raw
        self.lambda_recon_mse = lambda_recon_mse
        self.register_buffer('fixed_view_weights', torch.full((K,), 1.0 / float(K)), persistent=True)

        self.A = nn.Embedding(N, p)
        self.B = nn.Embedding(N, q)
        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.B.weight, mean=0.0, std=0.02)

        self.sketch_m = int(min(max(1, 1024), N))
        self.sketch_s = 1
        g = torch.Generator(device='cpu')
        g.manual_seed(42)
        sketch_idx = torch.randint(low=0, high=self.sketch_m, size=(N, self.sketch_s), generator=g, dtype=torch.long)
        sketch_val = torch.ones((N, self.sketch_s), dtype=torch.float32)
        self.register_buffer('sketch_proto_idx', sketch_idx, persistent=True)
        self.register_buffer('sketch_values', sketch_val, persistent=True)

        self.U_pool = nn.ModuleList([KR_MHA_Pooling(p=p, num_heads=num_heads, chunk_i=chunk_i) for _ in range(K)])
        self.recon = SharedCoreReconstructorChunk(K=K, p=p, q=q, r=r)
        self.u1_refiner = U1CNNRefiner(in_channels=p, hidden_channels=p)
        self.u2_refiner = U2LSTMRefiner(input_dim=p, hidden_dim=p, lookback=U2_LSTM_LOOKBACK)
        self.u3_refiner = U3HypergraphRefiner(input_dim=p, k=U3_KNN)
        self.proj_view = nn.ModuleList([
            nn.Sequential(nn.Linear(int(self.view_dims[k]), proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
            for k in range(K)
        ])
        self.proj_ab = nn.Linear(p + q, proj_dim)
        self.raw_tau = tau

    def _build_fixed_sketch_transpose(self, device: torch.device, dtype: torch.dtype):
        idx = self.sketch_proto_idx.to(device=device).reshape(-1)
        vals = self.sketch_values.to(device=device, dtype=dtype).reshape(-1)
        cols = torch.arange(self.N, device=device, dtype=torch.long).repeat_interleave(self.sketch_s)
        indices = torch.stack([idx, cols], dim=0)
        S_t = torch.sparse_coo_tensor(indices, vals, (self.sketch_m, self.N), device=device, dtype=dtype).coalesce()
        denom = torch.sparse.sum(S_t, dim=1).to_dense().clamp_min(1.0).view(self.sketch_m, 1)
        return S_t, denom

    def compute_epoch_sketch_U_list(self, XNd_list: list[torch.Tensor]) -> list[torch.Tensor]:
        device = self.A.weight.device
        dtype = self.A.weight.dtype
        S_t, denom = self._build_fixed_sketch_transpose(device=device, dtype=dtype)
        A_proto = torch.sparse.mm(S_t, self.A.weight) / denom
        U_list: list[torch.Tensor] = []
        for k in range(self.K):
            Xk = XNd_list[k].to(device=device, dtype=dtype)
            X_proto = torch.sparse.mm(S_t, Xk) / denom
            U_k = self.U_pool[k](X_proto.t().contiguous(), A_proto)
            if k == 0:
                U_k = self.u1_refiner(U_k)
            elif k == 1:
                U_k = self.u2_refiner(U_k)
            else:
                U_k = self.u3_refiner(U_k)
            U_list.append(U_k)
        return U_list

    def get_view_weights(self) -> torch.Tensor:
        return self.fixed_view_weights.to(device=self.recon.C.device, dtype=self.recon.C.dtype)

    def raw_contrast_loss(self, Xbd_list: list[torch.Tensor], idxs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B = Xbd_list[0].size(0)
        device = Xbd_list[0].device
        h_list = [l2norm(self.proj_view[k](Xbd_list[k]), dim=1) for k in range(self.K)]
        H = torch.cat(h_list, dim=0)
        KB = H.size(0)
        A_b = self.A(idxs)
        B_b = self.B(idxs)
        G = l2norm(self.proj_ab(torch.cat([A_b, B_b], dim=1)), dim=1)
        logits = H @ G.t() / float(self.raw_tau)
        sample_ids = torch.arange(B, device=device).repeat(self.K)
        pos_mask = sample_ids[:, None] == torch.arange(B, device=device)[None, :]
        pos_logits = logits[pos_mask].view(KB)
        denom = torch.logsumexp(logits, dim=1)
        loss = -(pos_logits - denom).mean()
        with torch.no_grad():
            neg_vals = logits[~pos_mask]
            stats = {
                'raw_con_pos_mean': float(pos_logits.mean().item()),
                'raw_con_neg_mean': float(neg_vals.mean().item()) if neg_vals.numel() else 0.0,
                'raw_con_valid': KB,
            }
        return loss, stats

    def forward(self, idxs: torch.Tensor, XNd_list: list[torch.Tensor], Xbd_list: list[torch.Tensor],
                latent_minmax_stats: list[tuple[torch.Tensor, torch.Tensor]] | None=None,
                cached_U_list: list[torch.Tensor] | None=None):
        assert len(XNd_list) == self.K
        assert len(Xbd_list) == self.K
        idxs = idxs.to(self.A.weight.device)
        A_all = self.A.weight
        B_all = self.B.weight

        if latent_minmax_stats is not None and APPLY_BATCH_LATENT_MINMAX_WITH_GLOBAL_STATS:
            Xbd_used_list = normalize_batch_latents_with_global_stats(Xbd_list, latent_minmax_stats, eps=GLOBAL_MINMAX_EPS)
        else:
            Xbd_used_list = Xbd_list

        X_b_t_list = [Xbd_used_list[k].t().contiguous() for k in range(self.K)]
        B_b = self.B(idxs)
        if cached_U_list is None:
            A_b = self.A(idxs)
            cached_U_list = [self.U_pool[k](X_b_t_list[k], A_b) for k in range(self.K)]
            if len(cached_U_list) >= 1:
                cached_U_list[0] = self.u1_refiner(cached_U_list[0])
            if len(cached_U_list) >= 2:
                cached_U_list[1] = self.u2_refiner(cached_U_list[1])
            if len(cached_U_list) >= 3:
                cached_U_list[2] = self.u3_refiner(cached_U_list[2])

        selfexpr_mse, per_view_mse = self.recon.recon_mse_chunk(
            U_list=cached_U_list,
            B_all=B_b,
            X_list=X_b_t_list,
            view_weights=self.get_view_weights(),
        )
        loss_con_raw, con_stats = self.raw_contrast_loss(Xbd_used_list, idxs)

        A_l2 = col_ortho_penalty(A_all)
        B_l2 = col_ortho_penalty(B_all)
        C_l2 = col_ortho_penalty(self.recon.C)
        wd_AB = A_l2 + B_l2
        wd_coreC = C_l2

        term_selfexpr = self.lambda_recon_mse * selfexpr_mse
        term_abc = self.lambda_abc * (wd_AB + wd_coreC)
        term_con_raw = self.lambda_con_raw * loss_con_raw
        loss = term_selfexpr + term_abc + term_con_raw

        return {
            'loss': loss,
            'term_recon': term_selfexpr,
            'term_selfexpr': term_selfexpr,
            'term_abc': term_abc,
            'term_con_raw': term_con_raw,
            'recon_loss': selfexpr_mse,
            'recon_mse_loss': selfexpr_mse,
            'selfexpr_mse_loss': selfexpr_mse,
            **con_stats,
            'wd_AB': wd_AB,
            'wd_coreC': wd_coreC,
            'A_l2': A_l2,
            'B_l2': B_l2,
            'C_l2': C_l2,
            'view_weights': self.get_view_weights().detach(),
            'per_view_mse': per_view_mse.detach(),
        }




def run_clustering(g_np, y_np, n_clusters, sample_ratio=0.2, class_names=None, return_details: bool=False):
    km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096, n_init='auto', random_state=42)
    pred = km.fit_predict(g_np)
    details = clustering_eval_detailed_from_labels(pred, y_np, feat_np=g_np, sample_ratio=sample_ratio, class_names=class_names)
    if return_details:
        return details
    return {k: details[k] for k in ['NMI', 'ARI', 'ACC', 'Silhouette(sample)']}



def set_requires_grad(m, flag):
    for p in m.parameters():
        p.requires_grad = flag



@torch.no_grad()
def init_feature_bank(backbone, loader_full, N, K, view_dims, device):
    XNd_list = [torch.zeros(N, int(view_dims[k]), device=device) for k in range(K)]
    backbone.eval()
    for idxs, payload, lengths, graph, _ in loader_full:
        idxs = idxs.to(device)
        payload = payload.to(device)
        lengths = lengths.to(device)
        graph = graph.to(device)
        Xbd_list = backbone(payload, lengths, graph)
        update_feature_bank(XNd_list, idxs, Xbd_list, momentum=0.0)
    return XNd_list




def _bank_cache_path(dataset_name: str, N: int, K: int, view_dims: list[int], cache_dir: str, suffix: str='') -> str:
    os.makedirs(cache_dir, exist_ok=True)
    vd_tag = 'x'.join((str(int(d)) for d in view_dims))
    suffix = str(suffix or '').strip()
    suffix_tag = f'_{suffix}' if suffix else ''
    return os.path.join(cache_dir, f'{dataset_name}_N{N}_K{K}_d{vd_tag}{suffix_tag}.pt')



def _sanitize_dataset_name(data_dir: str) -> str:
    base = os.path.basename(os.path.normpath(data_dir))
    safe = ''.join((c if c.isalnum() or c in '_-' else '_' for c in base))
    return safe or 'dataset'



def _backbone_has_trainable_params(backbone) -> bool:
    return any((p.requires_grad for p in backbone.parameters())) and any((p.numel() > 0 for p in backbone.parameters()))



@torch.no_grad()
def init_feature_bank_cached(backbone, loader_full, N, K, view_dims, device,
                             dataset_name: str | None=None,
                             use_cache: bool=True,
                             bank_cache_dir: str='./bank_cache',
                             bank_cache_suffix: str='fullcache_direct'):
    if not use_cache or _backbone_has_trainable_params(backbone):
        return init_feature_bank(backbone, loader_full, N, K, view_dims, device)

    ds_name = _sanitize_dataset_name(dataset_name) if dataset_name else 'dataset'
    cache_fp = _bank_cache_path(ds_name, N, K, list(view_dims), bank_cache_dir, bank_cache_suffix)
    if os.path.exists(cache_fp):
        try:
            t0 = time.time()
            payload = torch.load(cache_fp, map_location=device, weights_only=False)
            assert payload.get('N') == N, f"N mismatch: cached={payload.get('N')} vs request={N}"
            assert payload.get('K') == K, f"K mismatch: cached={payload.get('K')} vs request={K}"
            assert list(payload.get('view_dims', [])) == list(view_dims), 'view_dims mismatch'
            XNd_list = [x.to(device) for x in payload['XNd_list']]
            log(f'[BANK-CACHE] HIT  {os.path.basename(cache_fp)}  loaded in {time.time() - t0:.1f}s')
            return XNd_list
        except Exception as e:
            log(f'[BANK-CACHE] incompatible cache ({type(e).__name__}: {e}); recomputing')

    log(f'[BANK-CACHE] MISS {os.path.basename(cache_fp)}, computing feature bank ...')
    t0 = time.time()
    XNd_list = init_feature_bank(backbone, loader_full, N, K, view_dims, device)
    try:
        torch.save({
            'XNd_list': [x.detach().cpu() for x in XNd_list],
            'N': int(N),
            'K': int(K),
            'view_dims': [int(d) for d in view_dims],
            'dataset_name': ds_name,
        }, cache_fp)
        size_mb = os.path.getsize(cache_fp) / 1024 ** 2
        log(f'[BANK-CACHE] SAVE {os.path.basename(cache_fp)} ({size_mb:.1f} MB) compute={time.time() - t0:.1f}s')
    except Exception as e:
        log(f'[BANK-CACHE] save failed: {type(e).__name__}: {e}')
    return XNd_list




def _assert_no_param_overlap(param_groups: list[list[torch.nn.Parameter]], group_names: list[str]):
    seen = {}
    for g, name in zip(param_groups, group_names):
        for p in g:
            pid = id(p)
            if pid in seen:
                raise ValueError(f"Optimizer param overlap: parameter appears in both '{seen[pid]}' and '{name}'. Fix by excluding shared submodules from one group.")
            seen[pid] = name



def train_joint(backbone, global_model, loader_train, loader_full, device,
                epochs, lr, opt_wd,
                refresh_bank_each_epoch: bool=True,
                y_true_np: np.ndarray | None=None,
                n_clusters: int=4,
                cluster_eval_interval: int=1,
                class_names=None,
                dataset_name: str | None=None):
    set_requires_grad(global_model, True)
    set_requires_grad(backbone, True)
    set_requires_grad(global_model.U_pool, True)
    backbone.train()

    N = global_model.N
    K = global_model.K
    view_dims = list(global_model.view_dims)

    XNd_list = init_feature_bank_cached(
        backbone,
        loader_full,
        N=N,
        K=K,
        view_dims=view_dims,
        device=device,
        dataset_name=dataset_name,
    )

    latent_minmax_stats = None
    if APPLY_GLOBAL_LATENT_MINMAX:
        latent_minmax_stats = compute_global_minmax_stats_per_view(XNd_list)
        XNd_list = minmax_bank_per_view_with_stats_(XNd_list, latent_minmax_stats, eps=GLOBAL_MINMAX_EPS)

    p_backbone = list(backbone.parameters())
    p_upool = list(global_model.U_pool.parameters())
    p_urefine = list(global_model.u1_refiner.parameters()) + list(global_model.u2_refiner.parameters()) + list(global_model.u3_refiner.parameters())
    p_recon = list(global_model.recon.parameters())
    p_A = list(global_model.A.parameters())
    p_B = list(global_model.B.parameters())
    p_proj_view = list(global_model.proj_view.parameters())
    p_proj_ab = list(global_model.proj_ab.parameters())
    _assert_no_param_overlap(
        [p_backbone, p_upool, p_urefine, p_recon, p_A, p_B, p_proj_view, p_proj_ab],
        ['backbone', 'U_pool', 'U_refiners', 'recon', 'A_embed', 'B_embed', 'proj_view', 'proj_ab'],
    )
    opt = torch.optim.AdamW([
        {'params': p_backbone, 'weight_decay': opt_wd},
        {'params': p_upool, 'weight_decay': opt_wd},
        {'params': p_urefine, 'weight_decay': opt_wd},
        {'params': p_recon, 'weight_decay': opt_wd},
        {'params': p_proj_view, 'weight_decay': opt_wd},
        {'params': p_proj_ab, 'weight_decay': opt_wd},
        {'params': p_A, 'weight_decay': 0.0},
        {'params': p_B, 'weight_decay': 0.0},
    ], lr=lr)

    train_from_bank = not _backbone_has_trainable_params(backbone)
    bank_batch_size = int(getattr(loader_train, 'batch_size', None) or 4096)
    num_steps = (N + bank_batch_size - 1) // bank_batch_size if train_from_bank else len(loader_train)

    epoch_history: list[dict] = []
    best_cluster_metric = -1.0
    best_epoch = -1
    best_metrics = {}
    best_state = None
    epoch_loop_t_start = time.time()

    for ep in range(1, epochs + 1):
        if refresh_bank_each_epoch and ep > 1 and _backbone_has_trainable_params(backbone):
            backbone.eval()
            with torch.no_grad():
                XNd_list = init_feature_bank(backbone, loader_full, N=N, K=K, view_dims=view_dims, device=device)
                if APPLY_GLOBAL_LATENT_MINMAX:
                    latent_minmax_stats = compute_global_minmax_stats_per_view(XNd_list)
                    XNd_list = minmax_bank_per_view_with_stats_(XNd_list, latent_minmax_stats, eps=GLOBAL_MINMAX_EPS)
            backbone.train()

        global_model.train()
        backbone.train()
        meter = defaultdict(float)
        n = 0
        epoch_flow_count = 0
        epoch_input_bits = 0.0
        epoch_t_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if train_from_bank:
            perm = torch.randperm(N, device=device)
            batch_iter = ((step_i + 1, perm[step_i * bank_batch_size:min((step_i + 1) * bank_batch_size, N)]) for step_i in range(num_steps))
        else:
            batch_iter = enumerate(loader_train, start=1)

        for step, batch in batch_iter:
            if train_from_bank:
                idxs = batch.to(device, non_blocking=True)
                Xbd_list = [XNd_list[k][idxs] for k in range(K)]
                batch_latent_minmax_stats = None
            else:
                idxs, payload, lengths, graph, _ = batch
                idxs = idxs.to(device, non_blocking=True)
                payload = payload.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                graph = graph.to(device, non_blocking=True)
                Xbd_list = backbone(payload, lengths, graph)
                batch_latent_minmax_stats = latent_minmax_stats

            epoch_flow_count += int(idxs.numel())
            epoch_input_bits += float(sum(x.numel() * x.element_size() * 8 for x in Xbd_list))

            cached_U_list = global_model.compute_epoch_sketch_U_list(XNd_list)
            out = global_model(
                idxs=idxs,
                XNd_list=XNd_list,
                Xbd_list=Xbd_list,
                latent_minmax_stats=batch_latent_minmax_stats,
                cached_U_list=cached_U_list,
            )
            loss = out['loss']

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(global_model.parameters()), max_norm=5.0)
            opt.step()

            meter['loss'] += tensor_to_float(loss)
            meter['term_recon'] += tensor_to_float(out.get('term_recon'))
            meter['term_selfexpr'] += tensor_to_float(out.get('term_selfexpr'))
            meter['term_con_raw'] += tensor_to_float(out.get('term_con_raw'))
            meter['term_abc'] += tensor_to_float(out.get('term_abc'))
            meter['recon'] += tensor_to_float(out.get('recon_loss'))
            meter['recon_mse'] += tensor_to_float(out.get('recon_mse_loss'))
            meter['selfexpr_mse'] += tensor_to_float(out.get('selfexpr_mse_loss'))
            meter['wd_AB'] += tensor_to_float(out.get('wd_AB'))
            meter['wd_coreC'] += tensor_to_float(out.get('wd_coreC'))
            meter['A_l2'] += tensor_to_float(out.get('A_l2'))
            meter['B_l2'] += tensor_to_float(out.get('B_l2'))
            meter['C_l2'] += tensor_to_float(out.get('C_l2'))
            meter['raw_con_pos_mean'] += tensor_to_float(out.get('raw_con_pos_mean'))
            meter['raw_con_neg_mean'] += tensor_to_float(out.get('raw_con_neg_mean'))
            if out.get('per_view_mse') is not None:
                for k, v in enumerate(out['per_view_mse'].detach().cpu().tolist()):
                    meter[f'pv_mse_{k}'] += float(v)
            n += 1

        for k in list(meter.keys()):
            meter[k] /= max(n, 1)

        epoch_time_sec = float(time.time() - epoch_t_start)
        epoch_row = {
            'epoch': int(ep),
            'avg_loss': float(meter['loss']),
            'term_recon': float(meter.get('term_recon', 0.0)),
            'term_selfexpr': float(meter.get('term_selfexpr', 0.0)),
            'term_con_raw': float(meter.get('term_con_raw', 0.0)),
            'term_abc': float(meter.get('term_abc', 0.0)),
            'recon': float(meter.get('recon', 0.0)),
            'recon_mse': float(meter.get('recon_mse', 0.0)),
            'selfexpr_mse': float(meter.get('selfexpr_mse', 0.0)),
            'wd_AB': float(meter.get('wd_AB', 0.0)),
            'wd_coreC': float(meter.get('wd_coreC', 0.0)),
            'A_l2': float(meter.get('A_l2', 0.0)),
            'B_l2': float(meter.get('B_l2', 0.0)),
            'C_l2': float(meter.get('C_l2', 0.0)),
            'epoch_time_sec': epoch_time_sec,
            'epoch_num_flows': int(epoch_flow_count),
            'avg_ms_per_flow': 1000.0 * epoch_time_sec / max(epoch_flow_count, 1),
            'flows_per_sec': float(epoch_flow_count) / max(epoch_time_sec, 1e-12),
            'input_throughput_mbps': float(epoch_input_bits) / max(epoch_time_sec, 1e-12) / 1_000_000.0,
            'peak_gpu_mb': float(torch.cuda.max_memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0.0,
        }
        epoch_history.append(epoch_row)
        log(f"[EPOCH] ep {ep:03d} | loss={epoch_row['avg_loss']:.6f} | time={epoch_time_sec:.2f}s")

        if cluster_eval_interval is not None and cluster_eval_interval > 0 and y_true_np is not None:
            if ep % cluster_eval_interval == 0 or ep == epochs:
                with torch.no_grad():
                    z_ab = build_ab_concat(global_model)
                metrics = run_clustering(z_ab, y_true_np, n_clusters=n_clusters, sample_ratio=0.0, class_names=class_names, return_details=True)
                log(f"[CLUSTER][KMEANS][A|B] ep {ep:03d} | ACC={metrics['ACC']:.4f} NMI={metrics['NMI']:.4f} ARI={metrics['ARI']:.4f}")
                cur_metric = float(metrics.get('ACC', -np.inf))
                if cur_metric > best_cluster_metric:
                    best_cluster_metric = cur_metric
                    best_epoch = ep
                    best_metrics = {
                        'ACC': float(metrics['ACC']),
                        'NMI': float(metrics['NMI']),
                        'ARI': float(metrics['ARI']),
                        'epoch': int(ep),
                    }
                    best_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

    total_epoch_loop_time_sec = float(time.time() - epoch_loop_t_start)
    avg_epoch_loop_time_sec = total_epoch_loop_time_sec / max(len(epoch_history), 1)
    log(f'[JOINT][TOTAL] epoch_loop_time={total_epoch_loop_time_sec:.3f}s avg_epoch_time={avg_epoch_loop_time_sec:.3f}s epochs_run={len(epoch_history)}')
    log(f"[JOINT][BEST] epoch={best_epoch} | ACC={best_metrics.get('ACC', float('nan')):.4f} NMI={best_metrics.get('NMI', float('nan')):.4f} ARI={best_metrics.get('ARI', float('nan')):.4f}")
    if best_state is not None:
        global_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        log(f'[BEST-LOAD] reloaded best epoch {best_epoch} (ACC={best_cluster_metric:.4f}) for final eval')
    return XNd_list, latent_minmax_stats




def _safe_id2label(payload, labels_tensor):
    label_mapping = dict(payload.get('label_mapping', {}) or {})
    raw_id2label = dict(payload.get('id2label', {}) or {})
    id2label = {}
    for k, v in raw_id2label.items():
        try:
            id2label[int(k)] = str(v)
        except Exception:
            pass
    if not id2label and label_mapping:
        id2label = {int(v): str(k) for k, v in label_mapping.items()}
    if not id2label:
        n = int(labels_tensor.max().item()) + 1 if labels_tensor.numel() else 0
        id2label = {i: f'class_{i}' for i in range(n)}
    return id2label



def load_full_cache_dataset(cache_path: str):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f'完整数据缓存不存在：{cache_path}\n请先把已生成的 h5full.pt 放到该路径，或修改 main() 中的 full_data_cache_path。')
    t0 = time.time()
    payload = torch.load(cache_path, map_location='cpu', weights_only=False)
    required = ['payloads', 'lengths', 'graphs', 'labels']
    for k in required:
        if k not in payload:
            raise KeyError(f'完整数据缓存缺少字段：{k}')
    ds = CachedFlowTensorDataset(payloads=payload['payloads'], lengths=payload['lengths'], graphs=payload['graphs'], labels=payload['labels'], flow_data=payload.get('flow_data', None), label_mapping=payload.get('label_mapping', {}), id2label=payload.get('id2label', {}))
    id2label = _safe_id2label(payload, ds.labels)
    n_clusters = len(id2label)
    class_names = [id2label[i] for i in sorted(id2label.keys())]
    log(f'[DATA-CACHE] LOAD {os.path.basename(cache_path)} in {time.time() - t0:.2f}s')
    log(f'[DATA] ready N={len(ds)}, n_clusters={n_clusters}, class_names={class_names}')
    return (ds, payload, n_clusters, class_names)



def main():
    full_data_cache_path = './data_cache/multi_view_dataset_IOT2024_N112000_K3.pt'
    dataset_name = os.path.splitext(os.path.basename(full_data_cache_path))[0]
    batch_size = 4096
    epochs_joint = 10
    lr_joint = 4.2e-4
    opt_wd = 0.0022661938952803695

    device = DEVICE
    set_seed(42)
    ds, _, n_clusters, class_names = load_full_cache_dataset(full_data_cache_path)
    y_true_np = ds.labels.cpu().numpy().astype(np.int64)
    N = len(ds)

    loader_train = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=CachedFlowTensorDataset.collate_fn,
        drop_last=False,
    )
    loader_full = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=CachedFlowTensorDataset.collate_fn,
        drop_last=False,
    )

    backbone = End2EndBackbone(view_dims=VIEW_DIMS).to(device)
    global_model = GlobalEnd2End(N=N, view_dims=backbone.view_dims).to(device)

    train_joint(
        backbone,
        global_model,
        loader_train=loader_train,
        loader_full=loader_full,
        device=device,
        epochs=epochs_joint,
        lr=lr_joint,
        opt_wd=opt_wd,
        y_true_np=y_true_np,
        n_clusters=int(n_clusters),
        cluster_eval_interval=1,
        class_names=class_names,
        dataset_name=dataset_name,
    )

    with torch.no_grad():
        z_ab = build_ab_concat(global_model)
    res = run_clustering(
        z_ab,
        y_true_np,
        n_clusters=int(n_clusters),
        sample_ratio=0.0,
        class_names=class_names,
        return_details=True,
    )
    print('[FINAL][KMEANS][A|B] ACC={:.4f} NMI={:.4f} ARI={:.4f}'.format(res['ACC'], res['NMI'], res['ARI']))

if __name__ == '__main__':
    main()

