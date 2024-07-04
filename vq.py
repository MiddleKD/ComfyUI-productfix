import torch
from comfy import model_management
from diffusers.models.vq_model import VQModel

def load_vq_model(ckpt):
    """
    VQ 모델을 체크포인트에서 로드합니다. diffusers의 VQModel만 가능합니다.

    Args:
        ckpt (str): 체크포인트 파일 경로

    Returns:
        VQModel: 로드된 VQ 모델
    """
    offload_device: torch.device = model_management.intermediate_device()
    dtype = model_management.VAE_DTYPES[0]

    model_like = torch.load(ckpt, map_location="cpu")
    
    configs = model_like.get("configs", None)
    state_dict = model_like.get("state_dict", None)

    # 체크포인트 형식에 따라 다르게 로드
    if None in (configs, state_dict):
        vqmodel = VQModel.from_pretrained(ckpt)
    else:
        vqmodel = VQModel.from_config(configs)
        vqmodel.load_state_dict(state_dict)
    
    vqmodel.to(device=offload_device, dtype=dtype)

    return vqmodel

def vqmodel_encode(images, vqmodel:VQModel):
    """
    VQ 모델을 사용하여 이미지를 인코딩합니다.

    Args:
        images (torch.Tensor): 입력 이미지 텐서 (B, H, W, C)
        vqmodel (VQModel): VQ 모델

    Returns:
        torch.Tensor: 인코딩된 잠재 표현
    """
    device: torch.device = model_management.get_torch_device()
    offload_device: torch.device = model_management.intermediate_device()
    dtype = vqmodel.dtype

    vqmodel.to(device)
    
    # 이미지 전처리 및 인코딩
    images = (images*2-1).permute(0,3,1,2).to(device=device, dtype=dtype)
    latents = vqmodel.encode(images)["latents"]
    vqmodel.to(offload_device)

    return latents

def vqmodel_decode(latents, vqmodel:VQModel):
    """
    VQ 모델을 사용하여 잠재 표현을 디코딩합니다.

    Args:
        latents (torch.Tensor): 잠재 표현 텐서
        vqmodel (VQModel): VQ 모델

    Returns:
        torch.Tensor: 디코딩된 이미지 텐서 (B, H, W, C)
    """
    device: torch.device = model_management.get_torch_device()
    offload_device: torch.device = model_management.intermediate_device()
    dtype = vqmodel.dtype

    vqmodel.to(device)
    
    # 잠재 표현 디코딩 및 후처리
    images = vqmodel.decode(latents.to(device=device, dtype=dtype), force_not_quantize=True)["sample"]
    images = images * 0.5 + 0.5
    images = images.clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float()

    vqmodel.to(offload_device)

    return images