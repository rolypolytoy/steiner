import os
import random
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from borah_usc import build_rmst, total_wirelength, borah, draw_tree

script_dir = os.path.dirname(os.path.abspath(__file__))

#parameters
min_n = 3
max_n = 25
samples = 100

#necessary for formatting the output correctly.
print(f"{'deg':>4s}  {'avg_length':>10s}  {'time':>8s}  {'ms/inst':>8s}")
print(f"{'----':>4s}  {'----------':>10s}  {'--------':>8s}  {'--------':>8s}")

cumulative = 0.0

for n in range(min_n, max_n + 1):
    lengths = []
    last_tree = None
    last_terminals = None
    last_steiner = None

    t0 = time.time()
    for _ in range(samples):
        pts = [(random.random(), random.random()) for _ in range(n)]
        mst = build_rmst(pts)
        tree, steiner = borah(mst, max_passes=4)
        lengths.append(total_wirelength(tree))
        last_tree = tree
        last_terminals = pts
        last_steiner = steiner
    elapsed = time.time() - t0
    cumulative += elapsed
    ms = elapsed / samples * 1000
    avg = sum(lengths) / len(lengths)

    print(f"{n:4d}  {avg:10.4f}  {elapsed:7.2f}s  {ms:7.3f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    wl = total_wirelength(last_tree)
    draw_tree(ax, last_tree, last_terminals, last_steiner,
              last_terminals[0], f'Borah n={n}', wl, elapsed / samples)
    plt.tight_layout()
    plt.savefig(f"borah_n{n}.png", dpi=150, bbox_inches='tight')
    plt.close()

print(f"\nCumulative time: {cumulative:.2f}s")
