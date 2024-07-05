import logging
from functools import wraps
from comfy.samplers import KSamplerX0Inpaint

def inject_ksamplerx0inpaint_call(original_ksampler_call_fn):
    @wraps(original_ksampler_call_fn)
    def wrapper(self, x, sigma, denoise_mask, model_options={}, seed=None):
        
        # latent injection 옵션 확인
        latent_inject_options = model_options.get("is_latent_inject")
        if latent_inject_options is not None:
            start_sigma = latent_inject_options["start_sigma"]
            end_sigma = latent_inject_options["end_sigma"]
        
        if denoise_mask is not None:
            # 사용자 정의 마스크 함수 적용 (있는 경우)
            if "denoise_mask_function" in model_options:
                denoise_mask = model_options["denoise_mask_function"](sigma, denoise_mask, extra_options={"model": self.inner_model, "sigmas": self.sigmas})
            
            # latent injection 로직
            if latent_inject_options is not None:
                if sigma.item() > end_sigma and sigma.item() < start_sigma:
                    latent_mask = 1. - denoise_mask
                    scaled_latent = self.inner_model.inner_model.model_sampling.noise_scaling(sigma.reshape([sigma.shape[0]] + [1] * (len(self.noise.shape) - 1)), self.noise, self.latent_image)
                    x = x * denoise_mask + scaled_latent * latent_mask
            else:
                # 기본 inpainting 로직
                latent_mask = 1. - denoise_mask
                x = x * denoise_mask + self.inner_model.inner_model.model_sampling.noise_scaling(sigma.reshape([sigma.shape[0]] + [1] * (len(self.noise.shape) - 1)), self.noise, self.latent_image) * latent_mask
        
        # 모델 실행
        out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
        
        # 후처리 (latent injection이 아닌 경우에만)
        if denoise_mask is not None:
            if latent_inject_options is not None:
                pass
            else:
                out = out * denoise_mask + self.latent_image * latent_mask

        # 원래 함수로 복구
        if self.sigmas[-2].item() >= sigma.item():
            KSamplerX0Inpaint.__call__ = original_ksampler_call_fn
            logging.debug("\033[94m[middlek latent injection] KSamplerX0Inpaint.__call__ return to original_ksampler_call_fn\033[0m")

        return out

    return wrapper