"""Carga y muestreo del decoder CVAE condicionado del proyecto."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConditionalCVAEConfig:
    trajectory_length: int = 120
    n_assets: int = 2
    condition_dim: int = 14
    latent_dim: int = 8
    student_df: float = 5.0


class ConditionalCVAEDecoder:
    """Adaptador de inferencia para el decoder Keras guardado por el notebook."""

    def __init__(self, decoder: Any, config: ConditionalCVAEConfig) -> None:
        self.decoder = decoder
        self.config = config

    @classmethod
    def load(
        cls,
        decoder_path: str | Path,
        *,
        metadata_path: str | Path | None = None,
    ) -> "ConditionalCVAEDecoder":
        import keras

        decoder_path = Path(decoder_path)
        if not decoder_path.exists():
            raise FileNotFoundError(decoder_path)
        selected: dict[str, Any] = {}
        if metadata_path is not None:
            metadata_file = Path(metadata_path)
            if not metadata_file.exists():
                raise FileNotFoundError(metadata_file)
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            selected = metadata.get(
                "configuration", metadata.get("selected_config", {})
            )
        config = ConditionalCVAEConfig(
            latent_dim=int(selected.get("latent_dim", 8)),
            student_df=float(selected.get("student_df", 5.0)),
        )
        decoder = keras.models.load_model(decoder_path, compile=False, safe_mode=True)
        return cls(decoder, config)

    def sample(
        self,
        n: int,
        cond: np.ndarray,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        """Genera retornos normalizados con ruido Student-t bivariante."""

        if n <= 0:
            raise ValueError("n debe ser positivo")
        condition = np.asarray(cond, dtype=np.float32)
        if condition.ndim == 1:
            condition = condition[None, :]
        if condition.ndim != 2 or condition.shape[1] != self.config.condition_dim:
            raise ValueError(
                f"cond debe tener forma [batch, {self.config.condition_dim}]"
            )
        if condition.shape[0] == 1:
            condition = np.repeat(condition, n, axis=0)
        elif condition.shape[0] != n:
            raise ValueError("cond debe contener una fila o exactamente n filas")
        if not np.isfinite(condition).all():
            raise ValueError("cond contiene valores no finitos")

        rng = np.random.default_rng(seed)
        latent = rng.normal(size=(n, self.config.latent_dim)).astype(np.float32)
        raw_output = np.asarray(
            self.decoder(
                {"latent": latent, "condition": condition}, training=False
            )
        )
        location = raw_output[..., : self.config.n_assets]
        raw_scale = raw_output[
            ..., self.config.n_assets : 2 * self.config.n_assets
        ]
        scale = (
            np.log1p(np.exp(-np.abs(raw_scale)))
            + np.maximum(raw_scale, 0)
            + 1e-3
        )
        rho = 0.95 * np.tanh(raw_output[..., 2 * self.config.n_assets])

        independent = rng.normal(size=location.shape)
        correlated = np.empty_like(independent)
        correlated[..., 0] = independent[..., 0]
        correlated[..., 1] = (
            rho * independent[..., 0]
            + np.sqrt(np.maximum(1.0 - np.square(rho), 1e-6))
            * independent[..., 1]
        )
        chi_square = rng.chisquare(
            self.config.student_df, size=location.shape[:-1] + (1,)
        )
        student_noise = correlated / np.sqrt(chi_square / self.config.student_df)
        generated = location + scale * student_noise
        if generated.shape != (
            n,
            self.config.trajectory_length,
            self.config.n_assets,
        ):
            raise ValueError(f"Salida inesperada del decoder: {generated.shape}")
        if not np.isfinite(generated).all():
            raise ValueError("El CVAE ha generado valores no finitos")
        return generated.astype(np.float32)
