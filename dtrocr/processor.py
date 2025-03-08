from transformers import GPT2Tokenizer, AutoImageProcessor

from PIL import Image
from typing import List, Union
from config import DTrOCRConfig
from data import DTrOCRProcessorOutput
import torch
import numpy as np


class DTrOCRProcessor:
    def __init__(self, config: DTrOCRConfig, add_bos_token: bool = False, add_eos_token: bool = False):
        self.config = config
        self.vit_processor = AutoImageProcessor.from_pretrained(
            config.vit_hf_model,
            size={
                "height": config.image_size[0],
                'width': config.image_size[1]
            },
            use_fast=True
        )
        self.tokeniser = GPT2Tokenizer.from_pretrained(
            config.gpt2_hf_model,
            add_bos_token=add_bos_token,
            model_max_length=config.max_position_embeddings - int(
                (config.image_size[0] / config.patch_size[0]) * (config.image_size[1] / config.patch_size[1])
            )
        )
        self.tokeniser.pad_token = self.tokeniser.bos_token
        self.tokeniser.add_eos_token = add_eos_token

        # Bind a new method to gpt2_tokeniser
        self.tokeniser.build_inputs_with_special_tokens = modified_build_inputs_with_special_tokens.__get__(
            self.tokeniser
        )

        # DINO-specific image preprocessing (ImageNet normalization)
        self.mean = [0.485, 0.456, 0.406]  # ImageNet mean
        self.std = [0.229, 0.224, 0.225]  # ImageNet std

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess an image for DINO input."""
        # Resize to config.image_size
        image = image.resize((self.config.image_size[1], self.config.image_size[0]), Image.BILINEAR)
        # Convert to tensor and normalize
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0  # [C, H, W]
        for c in range(3):
            image[c] = (image[c] - self.mean[c]) / self.std[c]
        return image

    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]] = None,
        texts: Union[str, List[str]] = None,
        return_labels: bool = False,
        input_data_format: str = 'channels_last',
        padding: Union[bool, str] = False,
        *args,
        **kwargs
    ) -> DTrOCRProcessorOutput:
        text_inputs = self.tokeniser(
            texts, padding=padding, *args, **kwargs
        ) if texts is not None else None

        if images is not None:
            if isinstance(images, Image.Image):
                pixel_values = self.preprocess_image(images).unsqueeze(0)  # [1, C, H, W]
            else:
                pixel_values = torch.stack([self.preprocess_image(img) for img in images])  # [B, C, H, W]
        else:
            pixel_values = None

        return DTrOCRProcessorOutput(
            pixel_values=pixel_values,
            input_ids=text_inputs['input_ids'] if texts is not None else None,
            attention_mask=text_inputs['attention_mask'] if texts is not None else None,
            labels=text_inputs['input_ids'] if texts is not None and return_labels else None
        )


def modified_build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    if self.add_bos_token:
        bos_token_ids = [self.bos_token_id]
    else:
        bos_token_ids = []

    if self.add_eos_token:
        eos_token_ids = [self.eos_token_id]
    else:
        eos_token_ids = []

    output = bos_token_ids + token_ids_0 + eos_token_ids

    if token_ids_1 is None:
        return output

    return output + bos_token_ids + token_ids_1
