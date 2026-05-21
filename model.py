import math
import torch
import torch.nn as nn

#calculates wirelength of a given rectilinear edge sequence (RES)
def res(points, res_v, res_h):
    B, n, _ = points.shape
    x = points[:, :, 0]
    y = points[:, :, 1]
    lx, rx, ly, hy = x.clone(), x.clone(), y.clone(), y.clone()
    batch_idx = torch.arange(B, device=points.device)
    for i in range(n - 1):
        v, h = res_v[:, i], res_h[:, i]
        y_h, x_v = y[batch_idx, h], x[batch_idx, v]
        ly[batch_idx, v] = torch.min(ly[batch_idx, v], y_h)
        hy[batch_idx, v] = torch.max(hy[batch_idx, v], y_h)
        lx[batch_idx, h] = torch.min(lx[batch_idx, h], x_v)
        rx[batch_idx, h] = torch.max(rx[batch_idx, h], x_v)
    return ((hy - ly) + (rx - lx)).sum(dim=1)

#does reflection, 90 degree rotation or any combination, to get the 8 possible rotations from the paper
def transforms(points):
    x, y = points[..., 0], points[..., 1]
    transforms = []
    for swap in [False, True]:
        cx, cy = (y, x) if swap else (x, y)
        transforms.append(torch.stack([cx, cy], dim=-1))
        transforms.append(torch.stack([1.0 - cy, cx], dim=-1))
        transforms.append(torch.stack([1.0 - cx, 1.0 - cy], dim=-1))
        transforms.append(torch.stack([cy, 1.0 - cx], dim=-1))
    return transforms

#attention for transformers. self-explanatory
class TransformerBlock(nn.Module):
    def __init__(self, d=128, num_heads=16, ds=16, dh=512):
        super().__init__()
        self.num_heads, self.ds = num_heads, ds
        self.scale = 1.0 / math.sqrt(ds)
        self.W_Q = nn.Linear(d, num_heads * ds, bias=False)
        self.W_K = nn.Linear(d, num_heads * ds, bias=False)
        self.W_M = nn.Linear(d, num_heads * ds, bias=False)
        self.W_m = nn.Linear(num_heads * ds, d, bias=False)
        self.W1 = nn.Linear(d, dh)
        self.W2 = nn.Linear(dh, d)
        self.bn1 = nn.BatchNorm1d(d)
        self.bn2 = nn.BatchNorm1d(d)

    def forward(self, E):
        B, n, d = E.shape
        h, ds = self.num_heads, self.ds
        Q = self.W_Q(E).view(B, n, h, ds).transpose(1, 2)
        K = self.W_K(E).view(B, n, h, ds).transpose(1, 2)
        M = self.W_M(E).view(B, n, h, ds).transpose(1, 2)
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, M)
        mha_out = self.W_m(out.transpose(1, 2).contiguous().view(B, n, h * ds))
        E_mid = E + mha_out
        E_mid = self.bn1(E_mid.view(B * n, d)).view(B, n, d)
        ff_out = self.W2(torch.relu(self.W1(E_mid)))
        E_out = E_mid + ff_out
        return self.bn2(E_out.view(B * n, d)).view(B, n, d)

#main encoder, part of encoder-decoder transformer topology
class Encoder(nn.Module):
    def __init__(self, d=128, N=3, num_heads=16, ds=16, dh=512):
        super().__init__()
        self.d = d
        self.W_emb = nn.Linear(2, d, bias=False)
        self.bn_emb = nn.BatchNorm1d(d)
        self.layers = nn.ModuleList([TransformerBlock(d, num_heads, ds, dh) for _ in range(N)])

    def forward(self, V):
        B, n, _ = V.shape
        E = self.bn_emb(self.W_emb(V).view(B * n, self.d)).view(B, n, self.d)
        for layer in self.layers:
            E = layer(E)
        return E

#pointing mechanism from the paper
class Pointer(nn.Module):
    def __init__(self, d=128, dq=360, C=10.0, num_modes=1):
        super().__init__()
        self.C = C
        self.num_modes = num_modes
        self.W3 = nn.ModuleList([nn.Linear(d, dq, bias=False) for _ in range(num_modes)])
        self.W4 = nn.ModuleList([nn.Linear(dq, dq, bias=False) for _ in range(num_modes)])
        self.g = nn.ParameterList([nn.Parameter(torch.empty(dq)) for _ in range(num_modes)])
        for g in self.g:
            nn.init.uniform_(g, -1.0 / dq**0.5, 1.0 / dq**0.5)

    def forward(self, E, q, mask):
        logits_all = []
        for i in range(self.num_modes):
            logits = torch.tanh(self.W3[i](E) + self.W4[i](q).unsqueeze(1)) @ self.g[i]
            logits_all.append(logits)
        logits = torch.cat(logits_all, dim=-1)
        logits = self.C * torch.tanh(logits)
        if self.num_modes > 1:
            mask = torch.cat([mask] * self.num_modes, dim=-1)
        logits = logits.masked_fill(mask, float('-inf'))
        return torch.softmax(logits, dim=-1)

#decoder from transformer encoder-decoder topology
class Decoder(nn.Module):
    def __init__(self, d=128, dq=360, C=10.0):
        super().__init__()
        self.d, self.dq = d, dq
        self.ptr_u0 = Pointer(d, dq, C, num_modes=1)
        self.ptr_ut = Pointer(d, dq, C, num_modes=1)
        self.ptr_ws = Pointer(d, dq, C, num_modes=2)
        self.Wu = nn.Linear(d, dq, bias=False)
        self.Ww = nn.Linear(d, dq, bias=False)
        self.Wv = nn.Linear(d, dq, bias=False)
        self.Wh = nn.Linear(d, dq, bias=False)
        self.W_edge = nn.Linear(dq, dq, bias=False)
        self.W5 = nn.Linear(d, dq, bias=False)

    def _sample(self, probs, greedy):
        if greedy:
            return probs.argmax(dim=-1)
        safe = probs.clamp(min=0)
        safe = safe / safe.sum(dim=-1, keepdim=True)
        return torch.multinomial(safe, 1).squeeze(-1)

    def forward(self, E, greedy=False):
        B, n, d = E.shape
        device = E.device
        dq = self.dq
        batch_idx = torch.arange(B, device=device)
        res_v_list, res_h_list = [], []
        total_log_prob = torch.zeros(B, device=device)
        visited = torch.zeros(B, n, dtype=torch.bool, device=device)
        probs_u0 = self.ptr_u0(E, torch.zeros(B, dq, device=device),
        torch.zeros(B, n, dtype=torch.bool, device=device))
        u0 = self._sample(probs_u0, greedy)
        total_log_prob = total_log_prob + torch.log(probs_u0[batch_idx, u0] + 1e-30)
        visited = visited.scatter(1, u0.unsqueeze(1), True)
        edge_prev = torch.zeros(B, dq, device=device)
        subtree_prev = torch.zeros(B, dq, device=device)

        for t in range(1, n):
            q_t = torch.relu(edge_prev + subtree_prev)
            probs_ut = self.ptr_ut(E, q_t, visited)
            u_t = self._sample(probs_ut, greedy)
            total_log_prob = total_log_prob + torch.log(probs_ut[batch_idx, u_t] + 1e-30)
            e_ut = E[batch_idx, u_t]
            q_prime_t = torch.relu(edge_prev + subtree_prev + self.W5(e_ut))
            probs_ws = self.ptr_ws(E, q_prime_t, ~visited)
            ws_idx = self._sample(probs_ws, greedy)
            total_log_prob = total_log_prob + torch.log(probs_ws[batch_idx, ws_idx] + 1e-30)
            w_t, s_t = ws_idx % n, ws_idx // n
            v_t = torch.where(s_t == 0, u_t, w_t)
            h_t = torch.where(s_t == 0, w_t, u_t)
            res_v_list.append(v_t)
            res_h_list.append(h_t)
            visited = visited.scatter(1, u_t.unsqueeze(1), True)
            e_wt, e_vt, e_ht = E[batch_idx, w_t], E[batch_idx, v_t], E[batch_idx, h_t]
            edge_t = self.Wu(e_ut) + self.Ww(e_wt) + self.Wv(e_vt) + self.Wh(e_ht)
            subtree_prev = torch.max(subtree_prev, self.W_edge(edge_t))
            edge_prev = edge_t

        return torch.stack(res_v_list, dim=1), torch.stack(res_h_list, dim=1), total_log_prob

#encoder-decoder model, makes the decisions (policy network)
class Actor(nn.Module):
    def __init__(self, d=128, dq=360, C=10.0, N=3, num_heads=16, ds=16, dh=512):
        super().__init__()
        self.encoder = Encoder(d, N, num_heads, ds, dh)
        self.decoder = Decoder(d, dq, C)

    def forward(self, V, greedy=False):
        return self.decoder(self.encoder(V), greedy=greedy)

#critic network calculates expected wirelength (value network)
class Critic(nn.Module):
    def __init__(self, d=128, dc=256, N=3, num_heads=16, ds=16, dh=512):
        super().__init__()
        self.encoder = Encoder(d, N, num_heads, ds, dh)
        self.g_prime = nn.Parameter(torch.empty(d))
        nn.init.uniform_(self.g_prime, -1.0 / d**0.5, 1.0 / d**0.5)
        self.fc1 = nn.Linear(d, dc)
        self.fc2 = nn.Linear(dc, 1)

    def forward(self, V):
        E = self.encoder(V)
        weights = torch.softmax(torch.tanh(E) @ self.g_prime, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), E).squeeze(1)
        return self.fc2(torch.relu(self.fc1(context))).squeeze(-1)