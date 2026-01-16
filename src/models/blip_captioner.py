"""BLIP captioner for image captioning."""

import logging
from typing import List, Union
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class BLIPCaptioner:
    """Wrapper for BLIP model for image captioning."""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", device: torch.device = None):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logging.info(f"Loading BLIP captioning model: {model_name}")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def caption(self, images: Union[Image.Image, List[Image.Image]], max_length: int = 50) -> List[str]:
        if isinstance(images, Image.Image):
            images = [images]
            single_input = True
        else:
            single_input = False
        
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs, 
            max_length=max_length,
            min_length=5,
            num_beams=3,
            repetition_penalty=1.5
        )
        captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
        
        return captions[0] if single_input else captions
