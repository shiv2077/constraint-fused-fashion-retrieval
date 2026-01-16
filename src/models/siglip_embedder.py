"""SigLIP embedder for image and text encoding."""

import logging
from typing import List, Union
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

from src.common.utils import normalize_embeddings


class SigLIPEmbedder:
    """Wrapper for SigLIP model for image and text embedding."""
    
    def __init__(self, model_name: str = "google/siglip-so400m-patch14-384", device: torch.device = None):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logging.info(f"Loading SigLIP model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def encode_image(self, images: Union[Image.Image, List[Image.Image]], normalize: bool = True) -> np.ndarray:
        if isinstance(images, Image.Image):
            images = [images]
        
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model.get_image_features(**inputs)
        embeddings = outputs.cpu().numpy()
        
        if normalize:
            embeddings = normalize_embeddings(embeddings)
        
        return embeddings
    
    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model.get_text_features(**inputs)
        embeddings = outputs.cpu().numpy()
        
        if normalize:
            embeddings = normalize_embeddings(embeddings)
        
        return embeddings
