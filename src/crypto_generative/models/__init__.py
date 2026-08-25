"""Modelos generativos y baselines usados por el proyecto."""

from .block_bootstrap import (
    BlockBootstrapConfig,
    ConditionalBlockBootstrapConfig,
    ConditionalMultivariateBlockBootstrap,
    FROZEN_CONDITIONAL_BLOCK_LENGTH,
    FROZEN_CONDITIONAL_NEIGHBORS,
    MultivariateBlockBootstrap,
    frozen_conditional_bootstrap_config,
)

__all__ = [
    "BlockBootstrapConfig",
    "ConditionalBlockBootstrapConfig",
    "ConditionalCVAEConfig",
    "ConditionalCVAEDecoder",
    "ConditionalFlowConfig",
    "ConditionalFlowGenerator",
    "ConditionalRealNVP",
    "ConditionalGANConfig",
    "ConditionalGANGenerator",
    "ConditionalGenerator",
    "ConditionalMultivariateBlockBootstrap",
    "FROZEN_CONDITIONAL_BLOCK_LENGTH",
    "FROZEN_CONDITIONAL_NEIGHBORS",
    "FlowTrainingHistory",
    "MultivariateBlockBootstrap",
    "frozen_conditional_bootstrap_config",
]


def __getattr__(name: str):
    """Carga cada backend solo cuando se solicita."""
    cvae_names = {"ConditionalCVAEConfig", "ConditionalCVAEDecoder"}
    if name in cvae_names:
        from . import conditional_cvae

        return getattr(conditional_cvae, name)
    gan_names = {
        "ConditionalGANConfig",
        "ConditionalGANGenerator",
        "ConditionalGenerator",
    }
    if name in gan_names:
        from . import conditional_gan

        return getattr(conditional_gan, name)
    flow_names = {
        "ConditionalFlowConfig",
        "ConditionalFlowGenerator",
        "ConditionalRealNVP",
        "FlowTrainingHistory",
    }
    if name not in flow_names:
        raise AttributeError(name)
    from . import conditional_flow

    return getattr(conditional_flow, name)
