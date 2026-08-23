"""Arquitectura e inferencia de la GAN condicional BTC-ETH."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ConditionalGANConfig:
    trajectory_length: int = 120
    n_assets: int = 2
    condition_dim: int = 14
    latent_dim: int = 64
    generator_hidden_dim: int = 256
    discriminator_hidden_dim: int = 256
    generator_lr: float = 2e-4
    discriminator_lr: float = 1e-4
    batch_size: int = 128
    epochs: int = 80
    eval_every: int = 5
    real_label_smoothing: float = 0.90
    discriminator_dropout: float = 0.10
    grad_clip_norm: float = 5.0
    seed: int = 42

    @property
    def data_dim(self) -> int:
        return self.trajectory_length * self.n_assets


class ConditionalGenerator(nn.Module):
    """Generador MLP exactamente compatible con el checkpoint del notebook."""

    def __init__(self, config: ConditionalGANConfig) -> None:
        super().__init__()
        hidden = config.generator_hidden_dim
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(config.latent_dim + config.condition_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 2 * hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(2 * hidden, 2 * hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(2 * hidden, config.data_dim),
        )

    def forward(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([latent, condition], dim=-1))


class ConditionalGANGenerator:
    """Adaptador de inferencia con la interfaz común ``sample(n, cond)``."""

    def __init__(
        self,
        model: ConditionalGenerator,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device or self._auto_device())
        self.model = model.to(self.device).eval()
        self.config = model.config

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
    ) -> "ConditionalGANGenerator":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = ConditionalGANConfig(**checkpoint["selected_config"])
        model = ConditionalGenerator(config)
        model.load_state_dict(checkpoint["generator_state_dict"])
        return cls(model, device=device)

    def sample(
        self,
        n: int,
        cond: np.ndarray | torch.Tensor,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        if n <= 0:
            raise ValueError("n debe ser positivo")
        condition = torch.as_tensor(cond, dtype=torch.float32)
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)
        if condition.ndim != 2 or condition.shape[1] != self.config.condition_dim:
            raise ValueError(
                f"cond debe tener forma [batch, {self.config.condition_dim}]"
            )
        if condition.shape[0] == 1:
            condition = condition.repeat(n, 1)
        elif condition.shape[0] != n:
            raise ValueError("cond debe contener una fila o exactamente n filas")
        if not torch.isfinite(condition).all():
            raise ValueError("cond contiene valores no finitos")

        rng = torch.Generator(device="cpu").manual_seed(
            self.config.seed if seed is None else seed
        )
        latent = torch.randn(n, self.config.latent_dim, generator=rng).to(self.device)
        with torch.no_grad():
            generated = self.model(latent, condition.to(self.device))
        generated = generated.reshape(
            n, self.config.trajectory_length, self.config.n_assets
        ).cpu().numpy()
        if not np.isfinite(generated).all():
            raise ValueError("La GAN ha generado valores no finitos")
        return generated.astype(np.float32)
