import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np

import datasets
from positional_embeddings import PositionalEmbedding


#----------------------------------------------------------------------------
# Model Architecture

class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x


#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2). (independent design)
## https://github.com/NVlabs/edm/blob/main/generate.py#L25

@torch.no_grad()
def edm_sampler(
    edm, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, edm.sigma_min)
    sigma_max = min(sigma_max, edm.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([edm.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_hat = x_next
        t_hat = t_cur
        
        # Euler step.
        denoised = edm(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = edm(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


#----------------------------------------------------------------------------
# EDM model

class EDM():
    def __init__(self, model=None, cfg=None):
        self.cfg = cfg
        self.device = self.cfg.device
        self.model = model.to(self.device)
        ## parameters
        self.sigma_min = cfg.sigma_min
        self.sigma_max = cfg.sigma_max
        self.rho = cfg.rho
        self.sigma_data = cfg.sigma_data
        self.num_timesteps = cfg.num_timesteps
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5

    def model_forward_wrapper(self, x, sigma, **kwargs):
        """Wrapper for the model call"""
        sigma[sigma == 0] = self.sigma_min
        ## edm preconditioning for input and output
        ## https://github.com/NVlabs/edm/blob/main/training/networks.py#L632
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        model_output = self.model(torch.einsum('b,bk->bk', c_in, x), c_noise)
        return torch.einsum('b,bk->bk', c_skip, x) + torch.einsum('b,bk->bk', c_out, model_output)
        
    def train_step(self, images, labels=None, augment_pipe=None, **kwargs):
        ## https://github.com/NVlabs/edm/blob/main/training/loss.py#L66
        rnd_normal = torch.randn([images.shape[0]], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        noise = torch.randn_like(y)
        n = torch.einsum('b,bk->bk', sigma, noise)
        D_yn = self.model_forward_wrapper(y + n, sigma, labels=labels, augment_labels=augment_labels)
        loss = torch.einsum('b,bk->bk', weight, ((D_yn - y) ** 2))
        return loss.sum()
    
    def __call__(self, x, sigma, labels=None, augment_labels=None):
        if sigma.shape == torch.Size([]):
            sigma = sigma * torch.ones([x.shape[0]])
        return self.model_forward_wrapper(x.float(), sigma.float(), labels=labels, augment_labels=augment_labels)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="dino", choices=["circle", "dino", "line", "moons"])
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    # EDM models parameters
    parser.add_argument('--gt_guide_type', default='l2', type=str, help='gt_guide_type loss type')
    parser.add_argument('--sigma_min', default=0.002, type=float, help='sigma_min')
    parser.add_argument('--sigma_max', default=80.0, type=float, help='sigma_max')
    parser.add_argument('--rho', default=7., type=float, help='Schedule hyper-parameter')
    parser.add_argument('--sigma_data', default=0.5, type=float, help='sigma_data used in EDM for c_skip and c_out')
    parser.add_argument('--num_timesteps', default=50, type=int, help='timesteps for training')
    # Sampling parameters
    parser.add_argument('--total_steps', default=20, type=int, help='total_steps')
    parser.add_argument("--save_images_step", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    # Model architecture
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])
    
    config = parser.parse_args()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    config.device = device

    ## load dataset
    dataset = datasets.get_dataset(config.dataset)
    dataloader = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)

    ## init model
    mlp = MLP(
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        emb_size=config.embedding_size,
        time_emb=config.time_embedding,
        input_emb=config.input_embedding).to(device)
    
    edm = EDM(model=mlp, cfg=config)

    optimizer = torch.optim.AdamW(
        edm.model.parameters(),
        lr=config.learning_rate,
    )

    outdir = f"exps/{config.expr}"
    os.makedirs(outdir, exist_ok=True)
    global_step = 0
    frames = []
    losses = []
    print("Training model...")
    for epoch in range(config.num_epochs):
        edm.model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(dataloader):
            batch = batch[0].to(device)
            loss = edm.train_step(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(edm.model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()

        if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
            # generate data with the model to later visualize the learning process
            edm.model.eval()
            x_T = torch.randn([config.eval_batch_size, 2]).to(device).float()
            sample = edm_sampler(edm, x_T, num_steps=config.total_steps).detach().cpu()
            frames.append(sample.numpy())

    print("Saving model...")
    torch.save(edm.model.state_dict(), f"{outdir}/model.pth")

    print("Saving images...")
    imgdir = f"{outdir}/images"
    os.makedirs(imgdir, exist_ok=True)
    frames = np.stack(frames)
    xmin, xmax = -6, 6
    ymin, ymax = -6, 6
    for i, frame in enumerate(frames):
        plt.figure(figsize=(10, 10))
        plt.scatter(frame[:, 0], frame[:, 1])
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.savefig(f"{imgdir}/{i:04}.png")
        plt.close()

    print("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", np.array(losses))

    print("Saving frames...")
    np.save(f"{outdir}/frames.npy", frames)
