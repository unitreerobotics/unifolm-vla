"""
Framework factory utilities.
Selects and instantiates a registered framework implementation based on config.
"""

def build_framework(cfg):
    """
    Build a framework model from config.

    Args:
        cfg: Config object (OmegaConf / namespace) containing:
             cfg.framework.framework_py: Identifier string (e.g. "unifolm_vla")

    Returns:
        nn.Module: Instantiated framework model.

    Raises:
        NotImplementedError: If the specified framework id is unsupported.
    """
    if cfg.framework.framework_py == "unifolm_vla":
        from unifolm_vla.model.framework.unifolm_vla import Unifolm_VLA
        return Unifolm_VLA(cfg)
    
    raise NotImplementedError(f"Framework {cfg.framework.framework_py} is not implemented.")

