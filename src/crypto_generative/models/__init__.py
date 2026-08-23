"""Modelos generativos y baselines usados por el proyecto."""

from .block_bootstrap import BlockBootstrapConfig, MultivariateBlockBootstrap

__all__ = [
    "BlockBootstrapConfig",
    "ConditionalCVAEConfig",
    "ConditionalCVAEDecoder",
    "ConditionalFlowConfig",
    "ConditionalFlowGenerator",
    "ConditionalRealNVP",
    "ConditionalGANConfig",
    "ConditionalGANGenerator",
    "ConditionalGenerator",
    "FlowTrainingHistory",
    "MultivariateBlockBootstrap",
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
