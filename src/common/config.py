"""Configuration dataclasses for the fashion retrieval system."""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class IndexConfig:
    """Configuration for the indexing process."""
    model_name: str = "google/siglip-so400m-patch14-384"
    caption_model: str = "Salesforce/blip-image-captioning-base"
    max_images: Optional[int] = None
    image_size: int = 384
    batch_size: int = 8
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SearchConfig:
    """Configuration for the search/retrieval process."""
    model_name: str = "google/siglip-so400m-patch14-384"
    itm_model: str = "Salesforce/blip-itm-base-coco"
    
    # Search parameters
    topn: int = 50  # Number of candidates to retrieve
    topk: int = 5   # Number of final results to return
    
    # Scoring weights (tuned for high precision)
    w_vec: float = 0.20   # Vector similarity weight (reduced)
    w_itm: float = 0.30   # Image-text matching weight
    w_cons: float = 0.50  # Constraint satisfaction weight (INCREASED - most important)
    
    # Constraint penalty (strict enforcement)
    cons_penalty_threshold: float = 0.8   # Below 80% match = penalty
    cons_penalty_factor: float = 0.05     # 95% penalty if constraints don't match
    
    # Hard filtering
    require_all_constraints: bool = True  # Only return items matching ALL constraints
    
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchConfig":
        """Create from dictionary."""
        return cls(**data)


def save_config(config: IndexConfig | SearchConfig, path: Path) -> None:
    """Save configuration to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def load_index_config(path: Path) -> IndexConfig:
    """Load index configuration from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return IndexConfig.from_dict(data)


def load_search_config(path: Path) -> SearchConfig:
    """Load search configuration from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return SearchConfig.from_dict(data)
