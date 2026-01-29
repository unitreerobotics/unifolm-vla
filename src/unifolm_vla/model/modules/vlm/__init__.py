def get_vlm_model(config):
    
    vlm_name = config.framework.qwenvl.model_type
    if vlm_name == "qwen2_5_vl":
        from .QWen2_5 import _QWen_VL_Interface 
        return _QWen_VL_Interface(config)
    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")



