import os
import sys
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from model import Actor, res, transforms
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison"))
from borah_usc import build_rmst, borah, total_wirelength, draw_tree

checkpoint = "checkpoints/25.pt"
output_dir = "images"
t = 8

#runs a forward pass on the model
def infer(actor, V):
    with torch.no_grad():
        best_v, best_h, best_len = None, None, float('inf')
        for Vt in transforms(V)[:t]:
            rv, rh, _ = actor(Vt, greedy=True)
            l = res(V, rv, rh)[0].item()
            if l < best_len:
                best_v, best_h, best_len = rv[0], rh[0], l
    return best_v, best_h, best_len

#given a solution, generates a tree and plots it using matplotlib
def draw(V, best_v, best_h, best_len, name):
    pts = V[0].cpu()
    x, y = pts[:, 0].numpy().copy(), pts[:, 1].numpy().copy()
    n = len(x)
    lx, rx, ly, hy = x.copy(), x.copy(), y.copy(), y.copy()

    for i in range(n - 1):
        v, h = best_v[i].item(), best_h[i].item()
        ly[v], hy[v] = min(ly[v], y[h]), max(hy[v], y[h])
        lx[h], rx[h] = min(lx[h], x[v]), max(rx[h], x[v])

    steiner = set()
    for i in range(n):
        if hy[i] - ly[i] > 1e-9:
            for j in range(n):
                if rx[j] - lx[j] > 1e-9:
                    if lx[j]-1e-9 <= x[i] <= rx[j]+1e-9 and ly[i]-1e-9 <= y[j] <= hy[i]+1e-9:
                        steiner.add((round(x[i], 6), round(y[j], 6)))
    orig = set((round(x[i], 6), round(y[i], 6)) for i in range(n))
    steiner = [p for p in steiner if p not in orig]

    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(n):
        if hy[i] - ly[i] > 1e-9:
            ax.plot([x[i], x[i]], [ly[i], hy[i]], 'k-', lw=1.5)
        if rx[i] - lx[i] > 1e-9:
            ax.plot([lx[i], rx[i]], [y[i], y[i]], 'k-', lw=1.5)
    ax.scatter(x, y, c='#2563eb', s=70, marker='s', edgecolors='black', linewidths=0.5, zorder=3)
    if steiner:
        ax.scatter([p[0] for p in steiner], [p[1] for p in steiner],
                   c='#f97316', s=35, marker='D', edgecolors='black', linewidths=0.5, zorder=4)
    ax.set_title(f'REST  n={n}  L={best_len:.4f}  S={len(steiner)}')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150, bbox_inches='tight')
    plt.close()

#given the same points, runs borah's algorithm and saves an image
def bdraw(pts_list, name):
    terminals = [(float(x), float(y)) for x, y in pts_list]
    mst = build_rmst(terminals)
    tree, steiner = borah(mst, max_passes=4)
    wl = total_wirelength(tree)
    fig, ax = plt.subplots(figsize=(5, 5))
    draw_tree(ax, tree, terminals, steiner, terminals[0], f'Borah  n={len(terminals)}', wl, 0)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    return wl

#visualization function. saves generated steiner trees as images
def visualize(n, count):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    actor = Actor().to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()
    for idx in range(count):
        V = torch.rand(1, n, 2, device=device)
        best_v, best_h, best_len = infer(actor, V)
        draw(V, best_v, best_h, best_len, f"n{n}_{idx}")

#takes in custom points and benchmarks result
def benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    actor = Actor().to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()
    bench_dir = "benchmarks"
    if not os.path.isdir(bench_dir):
        return
    for fname in sorted(os.listdir(bench_dir)):
        if not fname.endswith(".txt"):
            continue
        pts = []
        with open(os.path.join(bench_dir, fname)) as f:
            for line in f:
                if len(pts) >= 25:
                    break
                line = line.strip()
                if not line:
                    break
                try:
                    x, y = map(float, line.split(','))
                    pts.append([x, y])
                except:
                    break
        if len(pts) < 3:
            continue
        name = os.path.splitext(fname)[0]
        V = torch.tensor(pts, dtype=torch.float32, device=device).unsqueeze(0)
        best_v, best_h, best_len = infer(actor, V)
        draw(V, best_v, best_h, best_len, f"{name}_rest")
        borah_wl = bdraw(pts, f"{name}_borah")
        print(f"{name}: REST={best_len:.4f}  Borah={borah_wl:.4f}")

if __name__ == '__main__':
    visualize(n=25, count=5)
    benchmark()