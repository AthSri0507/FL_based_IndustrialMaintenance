import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ClientEmbeddingGenerator(nn.Module):
    """Generator that outputs embeddings conditioned on client id."""

    def __init__(self, noise_dim: int, client_dim: int, emb_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + client_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, z: torch.Tensor, client_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, client_onehot], dim=1)
        return self.net(x)


class ClientEmbeddingDiscriminator(nn.Module):
    """Discriminator that scores embeddings conditioned on client id."""

    def __init__(self, emb_dim: int, client_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim + client_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, emb: torch.Tensor, client_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb, client_onehot], dim=1)
        return self.net(x).squeeze(-1)


def one_hot_client(ids: torch.Tensor, num_clients: int, device: Optional[torch.device] = None) -> torch.Tensor:
    if device is None:
        device = ids.device
    return F.one_hot(ids.long(), num_classes=num_clients).float().to(device)


def gan_train_step(
    gen: nn.Module,
    dis: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    real_emb: torch.Tensor,
    client_ids: torch.Tensor,
    num_clients: int,
    noise_dim: int = 32,
    device: Optional[torch.device] = None,
):
    """Single GAN training step (vanilla GAN BCE loss) on embedding space.

    - gen/dis operate on embeddings, not raw windows.
    - client_ids: tensor of shape (N,) with integer client indices.
    """
    if device is None:
        device = real_emb.device

    real_emb = real_emb.to(device)
    client_ids = client_ids.to(device)
    bsz = real_emb.size(0)

    # prepare client conditioning
    client_onehot = one_hot_client(client_ids, num_clients, device=device)

    # Discriminator step
    opt_d.zero_grad()
    # real scores
    real_logits = dis(real_emb, client_onehot)
    real_labels = torch.ones_like(real_logits)
    loss_fn = nn.BCEWithLogitsLoss()
    loss_d_real = loss_fn(real_logits, real_labels)

    # fake
    z = torch.randn(bsz, noise_dim, device=device)
    with torch.no_grad():
        fake_emb = gen(z, client_onehot)
    fake_logits = dis(fake_emb, client_onehot)
    fake_labels = torch.zeros_like(fake_logits)
    loss_d_fake = loss_fn(fake_logits, fake_labels)

    loss_d = (loss_d_real + loss_d_fake) * 0.5
    loss_d.backward()
    opt_d.step()

    # Generator step
    opt_g.zero_grad()
    z = torch.randn(bsz, noise_dim, device=device)
    fake_emb = gen(z, client_onehot)
    fake_logits = dis(fake_emb, client_onehot)
    loss_g = loss_fn(fake_logits, real_labels)
    loss_g.backward()
    opt_g.step()

    return {
        "loss_d": loss_d.item(),
        "loss_g": loss_g.item(),
    }

