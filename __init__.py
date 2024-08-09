import sys, os
sys.path.append(os.path.dirname(__file__))
from .node import (ResetModelPatcherCalculateWeight,
                   ApplyLatentInjection,
                   VQLoader,
                   VQEncoder,
                   VQDecoder,
                   GetTextMask,
                   DetailTransferAdd,
                   DetailTransferLatentAdd,
                   DynamicImageResize)


NODE_CLASS_MAPPINGS = {
    "ResetModelPatcherCalculateWeight": ResetModelPatcherCalculateWeight,
    "ApplyLatentInjection": ApplyLatentInjection,
    "VQLoader": VQLoader,
    "VQEncoder": VQEncoder,
    "VQDecoder": VQDecoder,
    "GetTextMask": GetTextMask,
    "DetailTransferAdd":DetailTransferAdd,
    "DetailTransferLatentAdd":DetailTransferLatentAdd,
    "DynamicImageResize":DynamicImageResize
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ResetModelPatcherCalculateWeight": "Reset injected model patcher (middlek)",
    "ApplyLatentInjection": "Apply latent injection (middlek)",
    "VQLoader": "VQ loader diffusers (middlek)",
    "VQEncoder": "VQ encoder diffusers (middlek)",
    "VQDecoder": "VQ decoder diffusers (middlek)",
    "GetTextMask": "Get text mask OCR (middlek)",
    "DetailTransferAdd":"Detail transfer mode:add (middlek)",
    "DetailTransferLatentAdd": "Detail transfer latent mode:add (middlek)",
    "DynamicImageResize":"Dynamic image resize (middlek)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
