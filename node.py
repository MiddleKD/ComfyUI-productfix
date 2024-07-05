import os
import logging
import folder_paths

from comfy import model_management
from comfy.model_patcher import ModelPatcher
from comfy.samplers import KSamplerX0Inpaint

from .advanced_sampler import inject_ksamplerx0inpaint_call
from .ocr import (get_text_mask, get_languages, language_map)
from .vq import (load_vq_model, vqmodel_encode, vqmodel_decode)
from .utils import (simple_resize, add_detail_transfer)

# ModelPatcher의 calculate_weight 메서드를 초기화하는 클래스
class ResetModelPatcherCalculateWeight:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model":("MODEL", ),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "reset_moodelpatcher_weight"
    CATEGORY = "productfix"

    def reset_moodelpatcher_weight(self, model:ModelPatcher):
        # 원래의 calculate_weight 메서드로 복원
        if hasattr(model, "original_calculate_weight"):
            model.calculate_weight = ModelPatcher.original_calculate_weight
            ModelPatcher.calculate_weight = ModelPatcher.original_calculate_weight
        return (model, )

# 잠재 공간에 이미지를 주입하는 클래스
class ApplyLatentInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model":("MODEL", ),
                             "latents": ("LATENT", ),
                             "inject_image_embed": ("LATENT", ),
                             "inject_mask": ("MASK",),
                             "start_sigma": ("FLOAT", {"default": 15.0}),
                             "end_sigma": ("FLOAT", {"default": 0.0})
                             }}
    RETURN_TYPES = ("MODEL", "LATENT",)
    FUNCTION = "apply_latent_injection"
    CATEGORY = "productfix"

    def apply_latent_injection(self, model, latents, inject_image_embed, inject_mask, start_sigma, end_sigma):
        device = model_management.get_torch_device()
        dtype = model_management.VAE_DTYPES[0]

        # KSamplerX0Inpaint의 __call__ 메서드를 수정된 버전으로 교체
        original_ksampler_call_fn = KSamplerX0Inpaint.__call__
        KSamplerX0Inpaint.__call__ = inject_ksamplerx0inpaint_call(original_ksampler_call_fn)
        logging.info("\033[94m[middlek latent injection] KSamplerX0Inpaint.__call__ is injected to inject_ksamplerx0inpaint_call\033[0m")

        # 잠재 공간 및 마스크 준비
        # (데이터 형식 변환 및 디바이스 이동)
        if isinstance(inject_image_embed, dict):
            inject_image_embed = inject_image_embed["samples"]
        
        inject_image_embed = inject_image_embed.to(dtype=dtype)
        b, c, h, w = inject_image_embed.shape
        if len(inject_mask.shape) != 4:
            inject_mask = inject_mask.unsqueeze(0)
        
        inject_image_embed = inject_image_embed.to(device=device, dtype=dtype)
        inject_mask = simple_resize(inject_mask, h, w).to(device=device, dtype=dtype)
        
        latents["samples"] = inject_image_embed
        latents["noise_mask"] = inject_mask

        # 모델 옵션에 latent injection 파라미터 추가
        if hasattr(model, "model_options"):
            model.model_options["is_latent_inject"] = {"start_sigma":start_sigma, "end_sigma":end_sigma}

        return (model, latents, )

# VQ 모델을 로드하는 클래스
class VQLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vq_name": ([os.path.basename(cur) for cur in folder_paths.get_filename_list("vae_approx") if "vq" in cur], ),
            }
        }

    RETURN_TYPES = ("VQ", )

    FUNCTION = "load_vq"
    CATEGORY = "productfix"

    def load_vq(self, vq_name):
        ckpt = os.path.join(folder_paths.get_folder_paths("vae_approx")[0], vq_name)
        vqmodel = load_vq_model(ckpt)
        return (vqmodel,)

# VQ 모델을 사용하여 이미지를 인코딩하는 클래스
class VQEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vq": ("VQ",),
                "images": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "encode"
    CATEGORY = "productfix"

    def encode(self, images, vq):
        latents = vqmodel_encode(images, vq)
        latents = {"samples":latents}
        return (latents,)

# VQ 모델을 사용하여 잠재 공간을 디코딩하는 클래스
class VQDecoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vq": ("VQ",),
                "latents": ("LATENT", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "decode"
    CATEGORY = "productfix"

    def decode(self, latents, vq):
        if isinstance(latents, dict):
            latents = latents.get("samples", None)
        images = vqmodel_decode(latents, vq)
        return (images,)

# 이미지에서 텍스트 마스크를 생성하는 클래스
class GetTextMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE",),
                    "languages": (
                        get_languages() + ["not use"],
                        {"default": "not use"},
                    ),
                    "codes": (
                        "STRING",
                        {"default": "en,ko", "multiline": False},
                    ),
                             }}
    RETURN_TYPES = ("MASK",)
    FUNCTION = "get_text_mask"
    CATEGORY = "productfix"

    def get_text_mask(self, image, languages:str, codes:str):
        # 언어 설정에 따라 타겟 언어 결정
        if languages != "not use":
            target_languages = [language_map[languages.split("/")[0]]]
        else:
            target_languages = codes.split(",")
        
        mask = get_text_mask(image, target_languages)
        return (mask, )

# 디테일 전송을 수행하는 클래스 (이미지 도메인)
class DetailTransferAdd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "target": ("IMAGE", ),
                    "source": ("IMAGE", ),
                    "blur": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.01}),
                    "blend_ratio": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001,  "round": 0.001}),
                },
            "optional": {
                "mask": ("MASK", ),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detail_transfer_add"
    CATEGORY = "productfix"

    def detail_transfer_add(self, target, source, blur, blend_ratio, mask=None):
        output_image = add_detail_transfer(target, source, blur, blend_ratio, mask)
        return (output_image, )

# 디테일 전송을 수행하는 클래스 (잠재 공간 도메인)
class DetailTransferLatentAdd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "target": ("LATENT", ),
                    "source": ("LATENT", ),
                    "blur": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.01}),
                    "blend_ratio": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001,  "round": 0.001}),
                },
            "optional": {
                "mask": ("MASK", ),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "detail_transfer_add"
    CATEGORY = "productfix"

    def detail_transfer_add(self, target, source, blur, blend_ratio, mask=None):
        # 잠재 공간을 이미지 형식으로 변환
        if type(target) == dict:
            target = target["samples"]
        if type(source) == dict:
            source = source["samples"]
        
        target = target.permute(0,2,3,1)
        source = source.permute(0,2,3,1)
        
        # 디테일 전송 수행
        output_image = add_detail_transfer(target, source, blur, blend_ratio, mask)
        
        # 결과를 다시 잠재 공간 형식으로 변환
        output_latent = output_image.permute(0,3,1,2)

        return (output_latent, )