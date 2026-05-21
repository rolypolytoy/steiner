import time
import torch
from model import Actor, res, transforms

#Parameters. Vary them based on what you want to test
checkpoint = "checkpoints/25.pt"
min_degree = 3
max_degree = 25
t = 8
num_tests = 10000
batch_size = 1000

#takes a checkpoint and samples from n = degree
def test_degree(actor, degree, device):
    total_length, total_count = 0.0, 0
    start = time.time()
    with torch.no_grad():
        while total_count < num_tests:
            B = min(batch_size, num_tests - total_count)
            V = torch.rand(B, degree, 2, device=device)
            if t == 1:
                rv, rh, _ = actor(V, greedy=True)
                lengths = res(V, rv, rh)
            else:
                best = torch.full((B,), float('inf'), device=device)
                for Vt in transforms(V)[:t]:
                    rv, rh, _ = actor(Vt, greedy=True)
                    best = torch.min(best, res(V, rv, rh))
                lengths = best
            total_length += lengths.sum().item()
            total_count += B
    elapsed = time.time() - start
    return total_length / total_count, elapsed, elapsed / total_count * 1000

#this function allows you to run the file directly rather than needing to specify params using CLI
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Sweep: degree {min_degree} to {max_degree}, T={t}, {num_tests} instances each\n")

    actor = Actor().to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()

    print(f"{'deg':>4s}  {'avg_length':>10s}  {'time':>8s}  {'ms/inst':>8s}")
    print(f"{'----':>4s}  {'----------':>10s}  {'--------':>8s}  {'--------':>8s}")

    total_start = time.time()
    total_instances = 0
    num_degrees = 0

    for degree in range(min_degree, max_degree + 1):
        avg, elapsed, ms = test_degree(actor, degree, device)
        print(f"{degree:4d}  {avg:10.4f}  {elapsed:7.2f}s  {ms:7.3f}")
        total_instances += num_tests
        num_degrees += 1

    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed:.4f}s")
    print(f"Total instances: {total_instances}")
