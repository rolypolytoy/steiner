"""
Borah, Owens, Irwin (1994) — Edge-Based Heuristic for Steiner Routing
======================================================================
Faithful O(n²) implementation of the algorithm described in the paper.

Key idea
--------
For every edge e1=(u,v) in the current MST, find the tree node p (other
than u and v) that gives the maximum gain when connected to the nearest
point on the *rectangular layout* of e1.  The nearest point p' on the
L-shaped layout of e1 to node p is just the rectilinear projection of p
onto the two axis-aligned segments that form e1's rectangular layout.

When p is connected to p', a loop is formed in the tree.  The longest
edge e2 on the tree path from p to whichever endpoint of e1 is farthest
from p is removed, and e1 itself is removed and replaced by the two
segments of its rectangular layout that go through p'.

    gain = length(e2) − manhattan(p, p')          [Eq. 1 in the paper]

The algorithm works in a *batched* mode:
  Pass:
    1. Compute the best (node, edge) pair for every edge  → O(n) pairs
    2. Sort pairs by gain descending
    3. Apply pairs greedily — skip a pair if either of its two edges has
       already been consumed in this pass
  Repeat until no positive-gain pair remains.

The paper reports that 3 passes suffice for virtually all inputs.

Dependencies: matplotlib (for plotting); everything else is stdlib.
"""

import argparse
import time
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────────────────────
# Graph — simple adjacency-list representation, no external dependencies
# ─────────────────────────────────────────────────────────────────────────────

class Graph:
    def __init__(self):
        self._nodes = set()
        self._adj   = {}          # node -> set of neighbours
        self._edges = {}          # canonical (u,v) tuple -> weight

    # ------------------------------------------------------------------
    def _key(self, u, v):
        return (u, v) if u <= v else (v, u)

    def add_node(self, n):
        self._nodes.add(n)
        self._adj.setdefault(n, set())

    def add_edge(self, u, v, weight):
        self.add_node(u)
        self.add_node(v)
        k = self._key(u, v)
        if k not in self._edges or self._edges[k] > weight:
            self._edges[k] = weight
            self._adj[u].add(v)
            self._adj[v].add(u)

    def remove_edge(self, u, v):
        k = self._key(u, v)
        if k in self._edges:
            del self._edges[k]
            self._adj[u].discard(v)
            self._adj[v].discard(u)

    def has_edge(self, u, v):
        return self._key(u, v) in self._edges

    def edge_weight(self, u, v):
        return self._edges[self._key(u, v)]

    def nodes(self):
        return list(self._nodes)

    def edges(self):
        return list(self._edges.keys())

    def neighbours(self, n):
        return list(self._adj.get(n, []))

    def copy(self):
        g = Graph()
        g._nodes = set(self._nodes)
        g._adj   = {n: set(nb) for n, nb in self._adj.items()}
        g._edges = dict(self._edges)
        return g


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def manhattan(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def total_wirelength(G):
    return sum(manhattan(u, v) for u, v in G.edges())


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Rectilinear MST via Prim's algorithm  (O(n²))
# ─────────────────────────────────────────────────────────────────────────────

def build_rmst(terminals):
    """Return a Graph that is the rectilinear MST of the terminals."""
    nodes = list(terminals)
    n = len(nodes)
    g = Graph()
    if n == 0:
        return g
    for nd in nodes:
        g.add_node(nd)
    if n == 1:
        return g

    in_tree     = {nodes[0]}
    not_in_tree = set(nodes[1:])

    while not_in_tree:
        best_w = float('inf')
        best_u = best_v = None
        for u in in_tree:
            for v in not_in_tree:
                w = manhattan(u, v)
                if w < best_w:
                    best_w, best_u, best_v = w, u, v
        g.add_edge(best_u, best_v, best_w)
        in_tree.add(best_v)
        not_in_tree.remove(best_v)

    return g


# ─────────────────────────────────────────────────────────────────────────────
# Nearest point on the rectangular layout of an edge
# ─────────────────────────────────────────────────────────────────────────────

def nearest_point_on_rect_layout(p, u, v):
    """
    The rectangular (L-shaped) layout of edge (u,v) consists of two
    axis-aligned segments that together form an L connecting u to v.
    There are two L-shapes; the nearest point p' to p is the same for
    both (it is the rectilinear projection of p onto the bounding
    rectangle of u and v).

    Concretely, p' = (clamp(p.x, min(u.x,v.x), max(u.x,v.x)),
                      clamp(p.y, min(u.y,v.y), max(u.y,v.y)))

    This is the unique point on any rectilinear path from u to v that
    minimises the Manhattan distance to p.
    """
    px, py = p
    x1, y1 = u
    x2, y2 = v
    cx = max(min(x1, x2), min(px, max(x1, x2)))
    cy = max(min(y1, y2), min(py, max(y1, y2)))
    return (cx, cy)


# ─────────────────────────────────────────────────────────────────────────────
# Tree path query: longest edge on the path between two nodes
# ─────────────────────────────────────────────────────────────────────────────

def longest_edge_on_path(tree, src, dst, forbidden_edge=None):
    """
    DFS from src to dst in the tree (which is acyclic).
    Returns (max_edge_weight, edge_u, edge_v) of the heaviest edge on
    the unique src→dst path, or None if dst is not reachable.

    forbidden_edge: optional (u,v) tuple — this edge is treated as absent
    during the search (used to prevent the path from crossing e1 itself).

    This mirrors the recursive DFS described in the paper (Section II,
    O(n²) implementation), where the 'maximum edge seen so far' is
    passed as a parameter along the recursion.
    """
    def is_forbidden(a, b):
        if forbidden_edge is None:
            return False
        fu, fv = forbidden_edge
        return (a == fu and b == fv) or (a == fv and b == fu)

    parent = {src: (None, None, 0)}   # node -> (parent_node, (eu,ev), weight)
    stack  = [src]

    while stack:
        node = stack.pop()
        if node == dst:
            break
        for nb in tree.neighbours(node):
            if nb in parent:
                continue
            if is_forbidden(node, nb):
                continue
            ew = tree.edge_weight(node, nb)
            parent[nb] = (node, (node, nb), ew)
            stack.append(nb)

    if dst not in parent:
        return None

    # Walk back from dst to src, find heaviest edge
    best_w  = -1
    best_ep = None
    cur = dst
    while parent[cur][0] is not None:
        par_node, ep, ew = parent[cur]
        if ew > best_w:
            best_w  = ew
            best_ep = ep
        cur = par_node

    return best_w, best_ep[0], best_ep[1]


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Compute all (node, edge) pairs with positive gain  [O(n²)]
# ─────────────────────────────────────────────────────────────────────────────

def compute_node_edge_pairs(tree):
    """
    For every edge e1=(u,v) in the tree, consider every other node p.
    Compute:
        p'   = nearest point on the rectangular layout of e1 to p
        e2   = longest edge on the tree path from p to the farther
               endpoint of e1  (the edge that will be 'cut')
        gain = length(e2) - manhattan(p, p')

    Per the paper, we report only the *best* (node, edge) pair for each
    edge (the one with maximum gain), because each edge can participate
    in at most one update per pass.

    Returns a list of dicts, one per edge that has a positive-gain node:
        { 'gain':  float,
          'node':  p,
          'p_prime': p',
          'e1': (u, v),       # the edge being 'entered' (will be removed)
          'e2': (eu, ev) }    # the longest path edge (will be removed)
    """
    all_nodes = tree.nodes()
    pairs = []

    for (u, v) in tree.edges():
        e1_w  = tree.edge_weight(u, v)
        best_gain = 0
        best_pair = None

        for p in all_nodes:
            if p == u or p == v:
                continue

            # Nearest point on the L-shaped layout of (u,v)
            p_prime = nearest_point_on_rect_layout(p, u, v)

            # If p_prime coincides with p, gain is always 0 (p is already
            # on the layout); skip.
            dist_p_pprime = manhattan(p, p_prime)

            # The loop formed by adding edge (p, p_prime) to the tree
            # runs from p → [tree path] → one of {u, v} → [part of e1] → p_prime.
            # The longest edge on the p-to-u and p-to-v tree paths is e2.
            # We pick whichever path gives the larger e2 (better gain).
            #
            # Note: the tree path from p to u does NOT include e1 itself,
            # so we query both endpoints and take the max.
            # Forbid e1 itself so the path doesn't shortcut across it
            res_u = longest_edge_on_path(tree, p, u, forbidden_edge=(u, v))
            res_v = longest_edge_on_path(tree, p, v, forbidden_edge=(u, v))

            e2_w  = -1
            e2_ep = None
            for res in (res_u, res_v):
                if res is not None and res[0] > e2_w:
                    e2_w  = res[0]
                    e2_ep = (res[1], res[2])

            if e2_w < 0:
                continue

            gain = e2_w - dist_p_pprime
            if gain > best_gain:
                best_gain = gain
                best_pair = {
                    'gain':    gain,
                    'node':    p,
                    'p_prime': p_prime,
                    'e1':      (u, v),
                    'e2':      e2_ep,
                }

        if best_pair is not None:
            pairs.append(best_pair)

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Apply edge-pair updates (batched pass)
# ─────────────────────────────────────────────────────────────────────────────

def apply_updates(tree, pairs):
    pairs_sorted = sorted(pairs, key=lambda x: -x['gain'])
    steiner_added = []
    remaining = list(pairs_sorted)

    while remaining:
        viable = [pr for pr in remaining
                  if tree.has_edge(*pr['e1']) and tree.has_edge(*pr['e2'])]
        if not viable:
            break

        pr = viable[0]
        remaining.remove(pr)
        u, v   = pr['e1']
        eu, ev = pr['e2']
        p      = pr['node']
        pp     = pr['p_prime']

        tree.remove_edge(u, v)
        tree.remove_edge(eu, ev)

        tree.add_node(pp)
        tree.add_edge(p,  pp, manhattan(p,  pp))
        tree.add_edge(pp, u,  manhattan(pp, u))
        tree.add_edge(pp, v,  manhattan(pp, v))

        steiner_added.append(pp)

    return steiner_added


# ─────────────────────────────────────────────────────────────────────────────
# Top-level Borah algorithm — repeated passes
# ─────────────────────────────────────────────────────────────────────────────

def borah(mst, max_passes=4):
    """
    Run the edge-based Steiner heuristic of Borah et al. 1994.

    Each pass:
      1. Compute best (node, edge) pair for every edge          [O(n²)]
      2. Sort pairs by gain descending                          [O(n log n)]
      3. Apply updates greedily (skip if edges already gone)    [O(n)]

    The paper shows 3 passes suffice for virtually all inputs.

    Returns (tree: Graph, all_steiner_points: list)
    """
    tree = mst.copy()
    all_steiner = []

    for pass_num in range(1, max_passes + 1):
        pairs = compute_node_edge_pairs(tree)
        # Keep only positive-gain pairs
        pairs = [pr for pr in pairs if pr['gain'] > 0]

        if not pairs:
            break

        added = apply_updates(tree, pairs)
        all_steiner.extend(added)
        wl = total_wirelength(tree)

        if not added:
            break

    return tree, all_steiner


# ─────────────────────────────────────────────────────────────────────────────
# Rectilinearize edges for visualization (L-shaped paths)
# ─────────────────────────────────────────────────────────────────────────────

def rectilinear_segments(tree):
    segs = []
    for u, v in tree.edges():
        x1, y1 = u
        x2, y2 = v
        if x1 == x2 or y1 == y2:
            segs.append((u, v))
        else:
            corner = (x1, y2)
            segs.append((u, corner))
            segs.append((corner, v))
    return segs


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def draw_tree(ax, tree, terminals, steiner, source, title, wl, runtime,
              rectilinearize=True):
    steiner_list = list(steiner)
    term_set     = set(map(tuple, terminals))

    segs = rectilinear_segments(tree) if rectilinearize else tree.edges()
    for (x1, y1), (x2, y2) in segs:
        ax.plot([x1, x2], [y1, y2], color='royalblue', linewidth=1.8, zorder=1)

    for p in terminals:
        color = 'tomato' if tuple(p) == tuple(source) else 'limegreen'
        ax.scatter(*p, s=60, color=color, zorder=5,
                   edgecolors='black', linewidths=0.8)

    for i, s in enumerate(steiner_list):
        ax.scatter(*s, s=50, color='gold', marker='D', zorder=5,
                   edgecolors='black', linewidths=0.8)
        ax.annotate(f'S{i+1}', xy=s, xytext=(4, 4),
                    textcoords='offset points', fontsize=7, color='saddlebrown')

    all_pts = list(terminals) + steiner_list
    min_x = min(p[0] for p in all_pts)
    min_y = min(p[1] for p in all_pts)
    max_x = max(p[0] for p in all_pts)
    max_y = max(p[1] for p in all_pts)
    pad = max(max_x - min_x, max_y - min_y) * 0.05 + 0.5

    ax.set_title(f'{title}\nWirelength: {wl}  |  Runtime: {runtime:.4f}s', fontsize=10)
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    src_patch  = mpatches.Patch(color='tomato',    label='Source node')
    term_patch = mpatches.Patch(color='limegreen', label='Terminal nodes')
    handles    = [src_patch, term_patch]
    if steiner_list:
        handles.append(mpatches.Patch(color='gold', label='Steiner points'))
    ax.legend(handles=handles, fontsize=7, loc='upper right')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Borah et al. 1994 — Edge-Based Steiner Routing')
    parser.add_argument('--input',  '-i', default='input_points.txt',
                        help='Input file with one "x y" point per line')
    parser.add_argument('--output', '-o', default='output.png',
                        help='Output PNG file')
    parser.add_argument('--passes', '-p', type=int, default=4,
                        help='Maximum number of improvement passes (default 4)')
    args = parser.parse_args()

    terminals = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            x, y = map(int, line.split())
            terminals.append((x, y))

    if len(terminals) < 2:
        sys.exit("Need at least 2 terminals.")


    # ── Step 1: RMST ────────────────────────────────────────
    t0 = time.perf_counter()
    mst = build_rmst(terminals)
    source  = terminals[0]
    mst_wl  = total_wirelength(mst)
    mst_time = time.perf_counter() - t0

    # ── Steps 2-4: Borah edge-based Steiner ─────────────────
    t1 = time.perf_counter()
    final_tree, steiner_added = borah(mst, max_passes=args.passes)
    final_wl   = total_wirelength(final_tree)
    borah_time = time.perf_counter() - t1

    improvement = mst_wl - final_wl
    pct = 100.0 * improvement / mst_wl if mst_wl else 0

    # ── Plot ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        'Rectilinear Steiner Minimum Tree — Borah et al. 1994 Edge-Based Heuristic',
        fontsize=13, fontweight='bold')

    draw_tree(axes[0], mst, terminals, [], source,
              'Initial RMST', mst_wl, mst_time, rectilinearize=False)
    draw_tree(axes[1], final_tree, terminals, steiner_added, source,
              'Final Steiner Routing (Borah 1994)', final_wl, borah_time)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
