"""Conditional RealNVP normalizing flow for joint BTC/ETH return paths.

The model learns an invertible transformation between a standard Gaussian base
and flattened return trajectories.  Every affine coupling layer is conditioned
on the 14-dimensional market-state vector prepared by the shared data pipeline.

Expected project shapes
-----------------------
condition: [batch, 14]
trajectory: [batch, 120, 2]

The public wrapper :class:`ConditionalFlowGenerator` exposes the common project
interface ``fit(X, cond)`` and ``sample(n, cond)`` while retaining exact model
log-likelihoods through ``log_prob``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class ConditionalFlowConfig:
    """Architecture and optimization settings for the conditional flow."""

    trajectory_length: int = 120
    n_assets: int = 2
    condition_dim: int = 14
    n_coupling_layers: int = 8
    hidden_dim: int = 256
    n_hidden_layers: int = 2
    scale_limit: float = 2.0
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 25
    min_delta: float = 1e-4
    grad_clip_norm: float = 5.0
    seed: int = 42

    @property
    def data_dim(self) -> int:
        return self.trajectory_length * self.n_assets


@dataclass
class FlowTrainingHistory:
    """Training trace returned by :meth:`ConditionalFlowGenerator.fit`."""

    train_nll: list[float] = field(default_factory=list)
    validation_nll: list[float] = field(default_factory=list)
    best_epoch: int = -1
    stopped_epoch: int = -1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _ConditionerMLP(nn.Module):
    """Predict affine shift and scale from the frozen coordinates + condition."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError("n_hidden_layers must be at least 1")

        layers: list[nn.Module] = []
        in_features = input_dim
        for _ in range(n_hidden_layers):
            layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.SiLU(),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim
        final = nn.Linear(in_features, output_dim)
        # Identity-like initialization stabilizes the first optimization steps.
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class ConditionalAffineCoupling(nn.Module):
    """Conditional affine coupling transform over a fixed partition."""

    def __init__(
        self,
        data_dim: int,
        condition_dim: int,
        identity_indices: Tensor,
        transform_indices: Tensor,
        hidden_dim: int,
        n_hidden_layers: int,
        scale_limit: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if identity_indices.numel() + transform_indices.numel() != data_dim:
            raise ValueError("Coupling partition must cover every data dimension")
        self.data_dim = data_dim
        self.scale_limit = float(scale_limit)
        self.register_buffer("identity_indices", identity_indices.long())
        self.register_buffer("transform_indices", transform_indices.long())
        self.conditioner = _ConditionerMLP(
            input_dim=identity_indices.numel() + condition_dim,
            output_dim=2 * transform_indices.numel(),
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            dropout=dropout,
        )

    def _shift_log_scale(self, x_identity: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        params = self.conditioner(torch.cat([x_identity, condition], dim=-1))
        shift, raw_log_scale = params.chunk(2, dim=-1)
        log_scale = self.scale_limit * torch.tanh(raw_log_scale / self.scale_limit)
        return shift, log_scale

    def forward(self, x: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        x_identity = x.index_select(-1, self.identity_indices)
        x_transform = x.index_select(-1, self.transform_indices)
        shift, log_scale = self._shift_log_scale(x_identity, condition)
        y_transform = x_transform * torch.exp(log_scale) + shift
        y = x.clone()
        y[:, self.identity_indices] = x_identity
        y[:, self.transform_indices] = y_transform
        log_abs_det = log_scale.sum(dim=-1)
        return y, log_abs_det

    def inverse(self, y: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        y_identity = y.index_select(-1, self.identity_indices)
        y_transform = y.index_select(-1, self.transform_indices)
        shift, log_scale = self._shift_log_scale(y_identity, condition)
        x_transform = (y_transform - shift) * torch.exp(-log_scale)
        x = y.clone()
        x[:, self.identity_indices] = y_identity
        x[:, self.transform_indices] = x_transform
        inverse_log_abs_det = -log_scale.sum(dim=-1)
        return x, inverse_log_abs_det


class FixedPermutation(nn.Module):
    """Deterministic feature permutation with zero Jacobian contribution."""

    def __init__(self, permutation: Tensor) -> None:
        super().__init__()
        permutation = permutation.long()
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel())
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", inverse)

    def forward(self, x: Tensor) -> Tensor:
        return x.index_select(-1, self.permutation)

    def inverse(self, y: Tensor) -> Tensor:
        return y.index_select(-1, self.inverse_permutation)


class ConditionalRealNVP(nn.Module):
    """Conditional RealNVP density model for flattened multivariate paths."""

    def __init__(self, config: ConditionalFlowConfig) -> None:
        super().__init__()
        self.config = config
        if config.data_dim % 2 != 0:
            raise ValueError("data_dim must be even for the half/half couplings")

        generator = torch.Generator().manual_seed(config.seed)
        transforms: list[nn.ModuleDict] = []
        half = config.data_dim // 2

        for layer_index in range(config.n_coupling_layers):
            permutation = torch.randperm(config.data_dim, generator=generator)
            if layer_index == 0:
                # Keep the first transform easy to inspect and reproduce.
                permutation = torch.arange(config.data_dim)
            identity = torch.arange(0, half)
            transform = torch.arange(half, config.data_dim)
            if layer_index % 2 == 1:
                identity, transform = transform, identity

            transforms.append(
                nn.ModuleDict(
                    {
                        "permutation": FixedPermutation(permutation),
                        "coupling": ConditionalAffineCoupling(
                            data_dim=config.data_dim,
                            condition_dim=config.condition_dim,
                            identity_indices=identity,
                            transform_indices=transform,
                            hidden_dim=config.hidden_dim,
                            n_hidden_layers=config.n_hidden_layers,
                            scale_limit=config.scale_limit,
                            dropout=config.dropout,
                        ),
                    }
                )
            )
        self.transforms = nn.ModuleList(transforms)

    def _validate_inputs(self, x: Tensor, condition: Tensor) -> None:
        if x.ndim != 2 or x.shape[-1] != self.config.data_dim:
            raise ValueError(f"x must have shape [batch, {self.config.data_dim}]")
        if condition.ndim != 2 or condition.shape[-1] != self.config.condition_dim:
            raise ValueError(
                f"condition must have shape [batch, {self.config.condition_dim}]"
            )
        if x.shape[0] != condition.shape[0]:
            raise ValueError("x and condition must have the same batch size")

    def transform_to_base(self, x: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        """Map observed trajectories ``x`` to Gaussian latent values ``z``."""

        self._validate_inputs(x, condition)
        z = x
        total_log_abs_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for transform in self.transforms:
            z = transform["permutation"](z)
            z, log_abs_det = transform["coupling"](z, condition)
            total_log_abs_det = total_log_abs_det + log_abs_det
        return z, total_log_abs_det

    def transform_from_base(self, z: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        """Map Gaussian latent values ``z`` to return trajectories ``x``."""

        self._validate_inputs(z, condition)
        x = z
        total_inverse_log_abs_det = torch.zeros(
            z.shape[0], device=z.device, dtype=z.dtype
        )
        for transform in reversed(self.transforms):
            x, inverse_log_abs_det = transform["coupling"].inverse(x, condition)
            x = transform["permutation"].inverse(x)
            total_inverse_log_abs_det = total_inverse_log_abs_det + inverse_log_abs_det
        return x, total_inverse_log_abs_det

    def log_prob(self, x: Tensor, condition: Tensor) -> Tensor:
        z, log_abs_det = self.transform_to_base(x, condition)
        base_log_prob = -0.5 * (z.square() + math.log(2.0 * math.pi)).sum(dim=-1)
        return base_log_prob + log_abs_det

    @torch.no_grad()
    def sample(self, condition: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        if condition.ndim != 2 or condition.shape[-1] != self.config.condition_dim:
            raise ValueError(
                f"condition must have shape [batch, {self.config.condition_dim}]"
            )
        if condition.device.type == "mps" and generator is not None:
            # MPS does not accept a CPU generator for a tensor created directly
            # on the MPS device. Generate reproducibly on CPU and transfer.
            z = torch.randn(
                condition.shape[0],
                self.config.data_dim,
                device="cpu",
                dtype=condition.dtype,
                generator=generator,
            ).to(condition.device)
        else:
            z = torch.randn(
                condition.shape[0],
                self.config.data_dim,
                device=condition.device,
                dtype=condition.dtype,
                generator=generator,
            )
        x, _ = self.transform_from_base(z, condition)
        return x


class ConditionalFlowGenerator:
    """High-level train/sample wrapper used by the workshop pipeline."""

    def __init__(
        self,
        config: ConditionalFlowConfig | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config or ConditionalFlowConfig()
        self.device = torch.device(device or self._auto_device())
        self._seed_everything(self.config.seed)
        self.model = ConditionalRealNVP(self.config).to(self.device)
        self.history = FlowTrainingHistory()
        self.is_fitted = False

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _prepare_trajectories(self, x: np.ndarray | Tensor) -> Tensor:
        tensor = torch.as_tensor(x, dtype=torch.float32)
        if tensor.ndim == 3:
            expected = (self.config.trajectory_length, self.config.n_assets)
            if tuple(tensor.shape[1:]) != expected:
                raise ValueError(f"X trailing shape must be {expected}, got {tuple(tensor.shape[1:])}")
            tensor = tensor.reshape(tensor.shape[0], -1)
        elif tensor.ndim != 2 or tensor.shape[1] != self.config.data_dim:
            raise ValueError(
                f"X must have shape [batch, {self.config.trajectory_length}, "
                f"{self.config.n_assets}] or [batch, {self.config.data_dim}]"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError("X contains NaN or infinite values")
        return tensor

    def _prepare_conditions(self, condition: np.ndarray | Tensor) -> Tensor:
        tensor = torch.as_tensor(condition, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != self.config.condition_dim:
            raise ValueError(
                f"cond must have shape [batch, {self.config.condition_dim}]"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError("cond contains NaN or infinite values")
        return tensor

    def _mean_nll_per_dimension(self, loader: DataLoader) -> float:
        self.model.eval()
        total_nll = 0.0
        total_examples = 0
        with torch.no_grad():
            for x_batch, c_batch in loader:
                x_batch = x_batch.to(self.device)
                c_batch = c_batch.to(self.device)
                nll = -self.model.log_prob(x_batch, c_batch)
                total_nll += nll.sum().item()
                total_examples += x_batch.shape[0]
        return total_nll / (total_examples * self.config.data_dim)

    def fit(
        self,
        X: np.ndarray | Tensor,
        cond: np.ndarray | Tensor,
        *,
        X_validation: np.ndarray | Tensor,
        cond_validation: np.ndarray | Tensor,
        verbose: bool = True,
    ) -> FlowTrainingHistory:
        """Fit the conditional flow by exact maximum likelihood.

        The validation arrays are mandatory so that architecture selection and
        early stopping never inspect the temporal test block.
        """

        x_train = self._prepare_trajectories(X)
        c_train = self._prepare_conditions(cond)
        x_val = self._prepare_trajectories(X_validation)
        c_val = self._prepare_conditions(cond_validation)
        if x_train.shape[0] != c_train.shape[0]:
            raise ValueError("X and cond must have the same number of samples")
        if x_val.shape[0] != c_val.shape[0]:
            raise ValueError("X_validation and cond_validation must align")

        train_generator = torch.Generator().manual_seed(self.config.seed)
        train_loader = DataLoader(
            TensorDataset(x_train, c_train),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=train_generator,
            drop_last=False,
        )
        validation_loader = DataLoader(
            TensorDataset(x_val, c_val),
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(3, self.config.patience // 4),
            min_lr=1e-5,
        )

        history = FlowTrainingHistory()
        best_state = copy.deepcopy(self.model.state_dict())
        best_val = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.config.max_epochs):
            self.model.train()
            total_train_nll = 0.0
            total_examples = 0
            for x_batch, c_batch in train_loader:
                x_batch = x_batch.to(self.device)
                c_batch = c_batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                nll = -self.model.log_prob(x_batch, c_batch).mean()
                loss = nll / self.config.data_dim
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite flow loss. Reduce learning_rate or scale_limit."
                    )
                loss.backward()
                if self.config.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                optimizer.step()
                total_train_nll += nll.item() * x_batch.shape[0]
                total_examples += x_batch.shape[0]

            train_nll = total_train_nll / (total_examples * self.config.data_dim)
            validation_nll = self._mean_nll_per_dimension(validation_loader)
            scheduler.step(validation_nll)
            history.train_nll.append(float(train_nll))
            history.validation_nll.append(float(validation_nll))

            improved = validation_nll < best_val - self.config.min_delta
            if improved:
                best_val = validation_nll
                best_state = copy.deepcopy(self.model.state_dict())
                history.best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if verbose and (epoch == 0 or (epoch + 1) % 5 == 0 or improved):
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch={epoch + 1:03d} "
                    f"train_nll_dim={train_nll:.6f} "
                    f"val_nll_dim={validation_nll:.6f} "
                    f"lr={current_lr:.2e}"
                )

            if epochs_without_improvement >= self.config.patience:
                history.stopped_epoch = epoch
                break
        else:
            history.stopped_epoch = self.config.max_epochs - 1

        self.model.load_state_dict(best_state)
        self.history = history
        self.is_fitted = True
        return history

    def log_prob(
        self,
        X: np.ndarray | Tensor,
        cond: np.ndarray | Tensor,
        *,
        per_dimension: bool = False,
        batch_size: int = 512,
    ) -> np.ndarray:
        """Return exact conditional log-density under the trained flow."""

        x = self._prepare_trajectories(X)
        c = self._prepare_conditions(cond)
        if x.shape[0] != c.shape[0]:
            raise ValueError("X and cond must have the same number of samples")
        loader = DataLoader(TensorDataset(x, c), batch_size=batch_size, shuffle=False)
        values: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for x_batch, c_batch in loader:
                log_p = self.model.log_prob(
                    x_batch.to(self.device), c_batch.to(self.device)
                )
                if per_dimension:
                    log_p = log_p / self.config.data_dim
                values.append(log_p.cpu().numpy())
        return np.concatenate(values)

    def sample(
        self,
        n: int,
        cond: np.ndarray | Tensor,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate ``n`` normalized BTC/ETH trajectories.

        ``cond`` may be one condition vector (repeated ``n`` times) or exactly
        ``n`` condition vectors (one generated path per state).
        """

        if n <= 0:
            raise ValueError("n must be positive")
        condition = self._prepare_conditions(cond)
        if condition.shape[0] == 1:
            condition = condition.repeat(n, 1)
        elif condition.shape[0] != n:
            raise ValueError("cond must contain one row or exactly n rows")
        condition = condition.to(self.device)

        generator_device = self.device.type if self.device.type != "mps" else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(self.config.seed if seed is None else seed)
        self.model.eval()
        with torch.no_grad():
            generated = self.model.sample(condition, generator=generator)
        return generated.reshape(
            n, self.config.trajectory_length, self.config.n_assets
        ).cpu().numpy()

    def save(self, path: str | Path, *, extra: dict[str, Any] | None = None) -> Path:
        """Persist architecture, weights and training trace in one checkpoint."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "model_state_dict": self.model.state_dict(),
            "history": self.history.to_dict(),
            "is_fitted": self.is_fitted,
            "extra": extra or {},
        }
        torch.save(payload, destination)
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
    ) -> "ConditionalFlowGenerator":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config_values = dict(checkpoint["config"])
        # Compatibilidad con los primeros checkpoints del notebook, que
        # llamaban ``epochs`` al límite denominado ``max_epochs`` en el módulo.
        if "epochs" in config_values and "max_epochs" not in config_values:
            config_values["max_epochs"] = config_values.pop("epochs")
        config = ConditionalFlowConfig(**config_values)
        generator = cls(config=config, device=device)
        generator.model.load_state_dict(checkpoint["model_state_dict"])
        generator.history = FlowTrainingHistory(**checkpoint.get("history", {}))
        generator.is_fitted = bool(checkpoint.get("is_fitted", True))
        return generator

    def export_metadata(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": "conditional_realnvp",
            "config": asdict(self.config),
            "history": self.history.to_dict(),
            "device_used": str(self.device),
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination
