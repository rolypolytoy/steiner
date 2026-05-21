import os
import torch
import torch.optim as optim
from model import Actor, Critic, res

#hyperparameters for training. lr and lr_decay are pulled straight from the paper
start_degree = 3
end_degree = 25
iterations = 10000
log_interval = 100
checkpoint_dir = "checkpoints"
initial_lr = 2.5e-4
lr_decay = 0.96

#rather than the 4096, 2048, 1024 and 512 approach the paper uses, we use a smoother reduction in batch size.
#the reason for this is my laptop 4070 has a small amount of VRAM and without this, it shut down mid-training
def get_batch_size(degree):
    tiers = [4096, 3072, 2560, 2048, 1536, 1024, 768, 512, 384, 256]
    midpoints = [(tiers[i] + tiers[i + 1]) / 2 for i in range(len(tiers) - 1)]
    ideal = 12288.0 / degree
    for i, t in enumerate(tiers):
        if i < len(midpoints) and ideal >= midpoints[i]:
            return t
    return tiers[-1]

#this is the main RL training loop using REINFORCE and the AdamW optimizer
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actor = Actor().to(device)
    critic = Critic().to(device)
    lr = initial_lr
    os.makedirs(checkpoint_dir, exist_ok=True)

    for degree in range(start_degree, end_degree + 1):
        print(f"\n{'='*60}\nTraining n = {degree}\n{'='*60}")

        #allows us to use  old checkpoints if training stopped. a must-have if you have low VRAM or poor battery life, it crashed frequently for me.
        if degree > start_degree:
            prev = os.path.join(checkpoint_dir, f"{degree - 1}.pt")
            if os.path.exists(prev):
                ckpt = torch.load(prev, map_location=device, weights_only=False)
                actor.load_state_dict(ckpt['actor'])
                critic.load_state_dict(ckpt['critic'])
                lr = ckpt['lr']

        lr *= lr_decay
        actor_opt = optim.AdamW(actor.parameters(), lr=lr, weight_decay = 1e-4)
        critic_opt = optim.AdamW(critic.parameters(), lr=lr, weight_decay = 1e-4)
        batch_size = get_batch_size(degree)
        print(f"LR: {lr:.6e} | Batch: {batch_size}")
        actor.train()
        critic.train()

        for it in range(1, iterations + 1):
            V = torch.rand(batch_size, degree, 2, device=device)
            res_v, res_h, log_prob = actor(V, greedy=False)
            with torch.no_grad():
                L = res(V, res_v, res_h)

            baseline = critic(V)
            advantage = baseline.detach() - L
            actor_loss = -(advantage * log_prob).mean()
            critic_loss = ((baseline - L.detach()) ** 2).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            actor_opt.step()
            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            critic_opt.step()

            if it % log_interval == 0:
                print(f"  {it:5d}/{iterations} | L: {L.mean().item():.4f} | "
                      f"B: {baseline.mean().item():.4f} | "
                      f"Adv: {advantage.mean().item():.4f} | "
                      f"AL: {actor_loss.item():.4f} | CL: {critic_loss.item():.4f}")

        torch.save({'degree': degree, 'actor': actor.state_dict(),
                    'critic': critic.state_dict(), 'lr': lr},
                   os.path.join(checkpoint_dir, f"{degree}.pt"))
