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
    # ... (INPUT_TYPES, RETURN_TYPES, FUNCTION, CATEGORY 정의)

    def reset_moodelpatcher_weight(self, model:ModelPatcher):
        # 원래의 calculate_weight 메서드로 복원
        if hasattr(model, "original_calculate_weight"):
            model.calculate_weight = ModelPatcher.original_calculate_weight
            ModelPatcher.calculate_weight = ModelPatcher.original_calculate_weight
        return (model, )

# 잠재 공간에 이미지를 주입하는 클래스
class ApplyLatentInjection:
    # ... (INPUT_TYPES, RETURN_TYPES, FUNCTION, CATEGORY 정의)

    def apply_latent_injection(self, model, latents, inject_image_embed, inject_mask, start_sigma, end_sigma):
        # KSamplerX0Inpaint의 __call__ 메서드를 수정된 버전으로 교체
        original_ksampler_call_fn = KSamplerX0Inpaint.__call__
        KSamplerX0Inpaint.__call__ = inject_ksamplerx0inpaint_call(original_ksampler_call_fn)
        logging.info("\033[94m[middlek image injection] KSamplerX0Inpaint.__call__ is injected to inject_ksamplerx0inpaint_call\033[0m")

        # 잠재 공간 및 마스크 준비
        # ... (데이터 형식 변환 및 디바이스 이동)

        # 모델 옵션에 이미지 주입 파라미터 추가
        if hasattr(model, "model_options"):
            model.model_options["is_latent_inject"] = {"start_sigma":start_sigma, "end_sigma":end_sigma}

        return (model, latents, )

# VQ 모델을 로드하는 클래스
class VQLoader:
    # ... (INPUT_TYPES, RETURN_TYPES, FUNCTION, CATEGORY 정의)

    def load_vq(self, vq_name):
        ckpt = os.path.join(folder_paths.get_folder_paths("vae_approx")[0], vq_name)
        vqmodel = load_vq_model(ckpt)
        return (vqmodel,)

# VQ 모델을 사용하여 이미지를 인코딩하는 클래스
class VQEncoder:
    # ... (INPUT_TYPES, RETURN_TYPES, FUNCTION, CATEGORY 정의)

    def encode(self, images, vq):
        latents = vqmodel_encode(images, vq)
        latents = {"samples":latents}
        return (latents,)

# VQ 모델을 사용하여 잠재 공간을 디코딩하는 클래스
class VQDecoder:
    # ... (INPUT_TYPES, RETURN_TYPES, FUNCTION, CATEGORY 정의)

    def decode(self, latents, vq):
        if isinstance(latents, dict):
            latents = latents.get("samples", None)
        images = vqmodel_decode(latents, vq)
        return (images,)

# 이미지에서 텍스트 마스크를 생성하는 클래스
class GetTextMask:
    # ... (INPUT_TYPES, RETURN_TYPES, FUNCTION, CATEGORY 정의)

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
    # ... (INPUT_TYPES, RETURN_TYPES, FUNCTION, CATEGORY 정의)

    def detail_transfer_add(self, target, source, blur, blend_ratio, mask=None):
        output_image = add_detail_transfer(target, source, blur, blend_ratio, mask)
        return (output_image, )

# 디테일 전송을 수행하는 클래스 (잠재 공간 도메인)
class DetailTransferLatentAdd:
    # ... (INPUT_TYPES, RETURN_TYPES, FUNCTION, CATEGORY 정의)

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