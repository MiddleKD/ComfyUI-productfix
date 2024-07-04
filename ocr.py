import os
from typing import List
import easyocr
import cv2
import torch
import numpy as np
import folder_paths

language_map = {
    "English": "en",
    "简体中文": "ch_sim",
    "繁體中文": "ch_tra",
    "العربية": "ar",
    "Azərbaycan": "az",
    "Euskal": "eu",
    "Bosanski": "bs",
    "Български": "bg",
    "Català": "ca",
    "Hrvatski": "hr",
    "Čeština": "cs",
    "Dansk": "da",
    "Nederlands": "nl",
    "Eesti": "et",
    "Suomi": "fi",
    "Français": "fr",
    "Galego": "gl",
    "Deutsch": "de",
    "Ελληνικά": "el",
    "עברית": "he",
    "हिन्दी": "hi",
    "Magyar": "hu",
    "Íslenska": "is",
    "Indonesia": "id",
    "Italiano": "it",
    "日本語": "ja",
    "한국어": "ko",
    "Latviešu": "lv",
    "Lietuvių": "lt",
    "Македонски": "mk",
    "Norsk": "no",
    "Polski": "pl",
    "Português": "pt",
    "Română": "ro",
    "Русский": "ru",
    "Српски": "sr",
    "Slovenčina": "sk",
    "Slovenščina": "sl",
    "Español": "es",
    "Svenska": "sv",
    "ไทย": "th",
    "Türkçe": "tr",
    "Українська": "uk",
    "Tiếng Việt": "vi",
}

def get_languages():
    """
    언어 맵에서 모든 언어를 "이름/코드" 형식의 문자열 리스트로 반환합니다.
    """
    return [f"{key}/{value}" for key, value in language_map.items()]

def get_text_mask(image: torch.Tensor, languages: List):
    """
    주어진 이미지에서 텍스트 영역을 감지하고 마스크를 생성합니다.

    Args:
        image (torch.Tensor): 입력 이미지 텐서 (B, H, W, C)
        languages (List): 감지할 언어 코드 리스트

    Returns:
        torch.Tensor: 텍스트 영역 마스크 (B, 1, H, W)
    """
    # EasyOCR 모델 디렉토리 설정
    model_dir = os.path.join(folder_paths.models_dir, "EasyOCR")
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    # 이미지 전처리
    b, h, w, c = image.shape
    image = image.to("cpu").numpy() * 255.0
    
    # EasyOCR 리더 초기화 및 텍스트 감지
    reader = easyocr.Reader(languages, model_storage_directory=model_dir)
    results = reader.readtext_batched(image, batch_size=b)
    
    text_masks = []
    for result in results:
        # 각 이미지에 대한 텍스트 마스크 생성
        text_mask = np.zeros([h,w,1], dtype=np.uint8)

        for detection in result:
            # 감지된 각 텍스트 영역을 마스크에 그리기
            bbox = detection[0]
            points = np.array(bbox).reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(text_mask, [points], 255)
        
        # 마스크를 텐서로 변환하고 정규화
        text_masks.append(torch.tensor(text_mask, dtype=torch.float32).reshape(1,h,w,1).permute(0,3,1,2) / 255.0)
    
    # 모든 마스크를 하나의 텐서로 결합
    text_masks = torch.cat(text_masks)

    return text_masks