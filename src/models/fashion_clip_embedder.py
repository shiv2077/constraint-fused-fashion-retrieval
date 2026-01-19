"""FashionCLIP embedder for fashion-specialized image and text encoding.

FashionCLIP is trained on 800K+ fashion images, providing much better
representations for fashion-specific attributes like colors, styles, and garments.
"""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Union, Optional
import numpy as np
import logging


class FashionCLIPEmbedder:
    """Fashion-specialized CLIP model for encoding images and text."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "patrickjohncyh/fashion-clip"
    ):
        """Initialize FashionCLIP model.
        
        Args:
            device: Device to use (cuda/cpu). Auto-detected if None.
            model_name: HuggingFace model name for FashionCLIP
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        
        logging.info(f"Loading FashionCLIP model: {model_name}")
        
        try:
            from fashion_clip.fashion_clip import FashionCLIP
            self.fclip = FashionCLIP('fashion-clip')
            self._use_fashion_clip = True
            logging.info("FashionCLIP loaded successfully")
        except Exception as e:
            logging.warning(f"Failed to load FashionCLIP: {e}")
            logging.warning("Falling back to standard CLIP")
            self._use_fashion_clip = False
            self._load_fallback_clip()
    
    def _load_fallback_clip(self):
        """Load standard CLIP as fallback."""
        from transformers import CLIPProcessor, CLIPModel
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(self.device)
        self.clip_model.eval()
    
    def encode_image(
        self,
        image: Union[Image.Image, List[Image.Image]],
        normalize: bool = True
    ) -> np.ndarray:
        """Encode image(s) to embedding vectors.
        
        Args:
            image: Single PIL Image or list of images
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            numpy array of shape (N, dim) where N is number of images
        """
        if isinstance(image, Image.Image):
            images = [image]
        else:
            images = image
        
        if self._use_fashion_clip:
            # FashionCLIP encoding
            embeddings = self.fclip.encode_images(images, batch_size=len(images))
            embeddings = np.array(embeddings)
        else:
            # Fallback CLIP encoding
            with torch.no_grad():
                inputs = self.clip_processor(
                    images=images, 
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                outputs = self.clip_model.get_image_features(**inputs)
                embeddings = outputs.cpu().numpy()
        
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def encode_text(
        self,
        text: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """Encode text(s) to embedding vectors.
        
        Args:
            text: Single string or list of strings
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            numpy array of shape (N, dim) where N is number of texts
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        if self._use_fashion_clip:
            # FashionCLIP encoding
            embeddings = self.fclip.encode_text(texts, batch_size=len(texts))
            embeddings = np.array(embeddings)
        else:
            # Fallback CLIP encoding
            with torch.no_grad():
                inputs = self.clip_processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(self.device)
                outputs = self.clip_model.get_text_features(**inputs)
                embeddings = outputs.cpu().numpy()
        
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def encode_text_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """Encode multiple texts in batches.
        
        Args:
            texts: List of strings to encode
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of shape (N, dim)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.encode_text(batch, normalize=normalize)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        if self._use_fashion_clip:
            return 512  # FashionCLIP uses 512-dim
        else:
            return 512  # CLIP ViT-B/32 also uses 512-dim
