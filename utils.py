import torch
from torchvision import transforms
from comfy import model_management

def simple_resize(image_tensor:torch.Tensor, height, width):
    """
    이미지 텐서를 지정된 높이와 너비로 단순 리사이즈합니다.
    """
    transform_resize = transforms.Resize((height, width))
    resized_tensor = transform_resize(image_tensor)
    return resized_tensor

def prepare_mask(mask, channels):
    """
    마스크를 지정된 채널 수에 맞게 확장합니다.
    """
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)   
    mask = mask.expand(-1, channels, -1, -1)
    return mask

def add_detail_transfer(target:torch.Tensor, source:torch.Tensor, blur:float, blend_ratio:float, mask:torch.Tensor=None):
    """
    타겟 이미지에 소스 이미지의 디테일을 전송합니다.

    Args:
        target (torch.Tensor): 타겟 이미지 텐서 (B, H, W, C)
        source (torch.Tensor): 소스 이미지 텐서 (B, H, W, C)
        blur (float): 가우시안 블러의 시그마 값
        blend_ratio (float): 블렌딩 비율
        mask (torch.Tensor, optional): 마스크 텐서

    Returns:
        torch.Tensor: 디테일이 전송된 결과 이미지 텐서
    """
    # 텐서 shape 및 디바이스 설정
    B, H, W, C = target.shape
    device = model_management.get_torch_device()
    target_tensor = target.permute(0, 3, 1, 2).clone().to(device)
    source_tensor = source.permute(0, 3, 1, 2).clone().to(device)

    # 소스 이미지 리사이즈 (필요한 경우)
    if target.shape[2:] != source.shape[2:]:
        source_tensor = simple_resize(source_tensor, H, W)

    # 배치 크기 맞추기 (필요한 경우)
    if source.shape[0] < B:
        source = source[0].unsqueeze(0).repeat(B, 1, 1, 1)

    # 가우시안 블러 설정 및 적용
    kernel_size = int(6 * int(blur) + 1)
    gaussian_blur = transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(blur, blur))
    blurred_target = gaussian_blur(target_tensor)
    blurred_source = gaussian_blur(source_tensor)
    
    # 디테일 전송 수행
    tensor_out = (source_tensor - blurred_source) + blurred_target
    tensor_out = torch.lerp(target_tensor, tensor_out, blend_ratio)

    # 마스크 적용 (있는 경우)
    if mask is not None:
        mask = prepare_mask(mask, C)
        mask = mask.to(device)
        if target.shape[2:] != mask.shape[2:]:
            mask = simple_resize(mask, H, W)
        tensor_out = torch.lerp(target_tensor, tensor_out, mask)

    # 결과 클램핑 및 형식 변환
    tensor_out = torch.clamp(tensor_out, 0, 1)
    tensor_out = tensor_out.permute(0, 2, 3, 1).cpu().float()

    return tensor_out