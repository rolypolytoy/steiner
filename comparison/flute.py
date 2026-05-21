import subprocess
import os
import time
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
flute_bin = os.path.join(script_dir, "flute", "build", "flute-net")
flute_dir = script_dir

#do not modify scale. since the FLUTE code takes a 100x100 integer grid as its input, this scale factor is what allows us to convert between a [0,100]^2 and a [0,1]^2 grid
scale = 10000
#feel free to modify these parameters
a = 3
min_n = 3
max_n = 25
samples = 1000

def flute_length(pts):
    input_str = "\n".join(f"{int(x * scale)} {int(y * scale)}" for x, y in pts)
    result = subprocess.run([flute_bin, str(a)], input=input_str, capture_output=True, text=True, cwd=flute_dir)
    parts = result.stdout.strip().split()
    wl = int(parts[0]) / scale
    us = float(parts[1])
    return wl, us

print(f"{'deg':>4s}  {'avg_length':>10s}  {'avg_us':>8s}")
print(f"{'----':>4s}  {'----------':>10s}  {'--------':>8s}")

#outputs FLUTE degree/length/timing
cumulative_us = 0.0
for n in range(min_n, max_n + 1):
    lengths = []
    times_us = []
    for _ in range(samples):
        pts = [(random.random(), random.random()) for _ in range(n)]
        wl, us = flute_length(pts)
        lengths.append(wl)
        times_us.append(us)
    avg_len = sum(lengths) / len(lengths)
    avg_us = sum(times_us) / len(times_us)
    cumulative_us += sum(times_us)
    print(f"{n:4d}  {avg_len:10.4f}  {avg_us:8.3f}")

print(f"\nCumulative FLUTE time: {cumulative_us / 1e6:.4f}s")
