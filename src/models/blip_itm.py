"""BLIP ITM (Image-Text Matching) scorer for cross-modal reranking."""

import logging
from typing import List, Tuple
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval
import numpy as np


class BLIPITM:
    """Wrapper for BLIP ITM model for image-text matching."""
    
    def __init__(self, model_name: str = "Salesforce/blip-itm-base-coco", device: torch.device = None):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logging.info(f"Loading BLIP ITM model: {model_name}")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForImageTextRetrieval.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def score(self, images: List[Image.Image], text: str) -> np.ndarray:
        if not images:
            return np.array([])
        
        texts = [text] * len(images)
        
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        
        # Get the ITM logits and convert to probabilities
        # ITM head outputs logits for [not_match, match]
        itm_logits = outputs.itm_score
        probs = torch.softmax(itm_logits, dim=1)
        match_probs = probs[:, 1]  # Probability of match
        
        return match_probs.cpu().numpy()
    
    @torch.no_grad()
    def score_batch(self, image_text_pairs: List[Tuple[Image.Image, str]], batch_size: int = 8) -> np.ndarray:
        all_scores = []
        
        for i in range(0, len(image_text_pairs), batch_size):
            batch = image_text_pairs[i:i + batch_size]
            images = [pair[0] for pair in batch]
            texts = [pair[1] for pair in batch]
            
            inputs = self.processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            itm_logits = outputs.itm_score
            probs = torch.softmax(itm_logits, dim=1)
            match_probs = probs[:, 1]
            
            all_scores.append(match_probs.cpu().numpy())
        
        return np.concatenate(all_scores) if all_scores else np.array([])
