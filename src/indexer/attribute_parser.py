"""Attribute parser for extracting structured tags from captions and queries."""

import re
from typing import Dict, List, Set


# Fashion vocabulary with synonyms
COLOR_VOCAB = {
    'red', 'crimson', 'scarlet', 'burgundy', 'maroon',
    'blue', 'navy', 'azure', 'cobalt', 'indigo', 'turquoise', 'cyan',
    'green', 'emerald', 'lime', 'olive', 'teal', 'mint',
    'yellow', 'gold', 'golden', 'amber', 'mustard',
    'orange', 'coral', 'peach', 'tangerine',
    'purple', 'violet', 'lavender', 'plum', 'magenta',
    'pink', 'rose', 'fuchsia', 'salmon',
    'brown', 'tan', 'beige', 'khaki', 'chocolate', 'coffee',
    'black', 'white', 'gray', 'grey', 'silver',
    'cream', 'ivory', 'off-white',
    'multicolor', 'multi-color', 'colorful', 'patterned',
}

GARMENT_VOCAB = {
    # Tops
    'shirt', 'blouse', 't-shirt', 'tshirt', 'tee', 'top', 'tank', 'tank top',
    'sweater', 'cardigan', 'hoodie', 'sweatshirt', 'pullover',
    'jacket', 'coat', 'blazer', 'vest', 'windbreaker', 'parka',
    
    # Bottoms
    'pants', 'trousers', 'jeans', 'shorts', 'skirt', 'leggings',
    'dress', 'gown', 'frock', 'sundress',
    
    # Full body
    'suit', 'jumpsuit', 'romper', 'overalls',
    
    # Accessories
    'hat', 'cap', 'beanie', 'scarf', 'gloves', 'belt',
    'bag', 'purse', 'backpack', 'handbag', 'tote',
    'shoes', 'boots', 'sneakers', 'sandals', 'heels', 'flats',
    'sunglasses', 'glasses', 'watch', 'bracelet', 'necklace', 'earrings',
    'tie', 'bowtie',
}

CONTEXT_VOCAB = {
    # Occasions
    'casual', 'formal', 'business', 'professional', 'elegant',
    'party', 'wedding', 'evening', 'cocktail', 'gala',
    'sports', 'athletic', 'gym', 'workout', 'running', 'yoga',
    'beach', 'summer', 'winter', 'fall', 'spring', 'autumn',
    'outdoor', 'indoor', 'vacation', 'travel',
    
    # Styles
    'vintage', 'retro', 'modern', 'contemporary', 'classic',
    'bohemian', 'boho', 'hipster', 'preppy', 'streetwear',
    'minimalist', 'chic', 'trendy', 'fashionable',
    
    # Settings
    'office', 'work', 'school', 'university', 'home',
    'restaurant', 'dinner', 'lunch', 'brunch',
    'concert', 'festival', 'club', 'bar',
    'hiking', 'camping', 'cycling', 'skiing',
}


def extract_tags(caption: str) -> Dict[str, Set[str]]:
    caption_lower = caption.lower()
    
    colors = set()
    for color in COLOR_VOCAB:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(color) + r'\b'
        if re.search(pattern, caption_lower):
            colors.add(color)
    
    # Extract garments
    garments = set()
    for garment in GARMENT_VOCAB:
        # Handle multi-word garments like "tank top"
        pattern = r'\b' + re.escape(garment) + r'\b'
        if re.search(pattern, caption_lower):
            garments.add(garment)
    
    # Extract contexts
    contexts = set()
    for context in CONTEXT_VOCAB:
        pattern = r'\b' + re.escape(context) + r'\b'
        if re.search(pattern, caption_lower):
            contexts.add(context)
    
    return {
        'colors': colors,
        'garments': garments,
        'contexts': contexts,
    }


def parse_query_constraints(query: str) -> Dict[str, Set[str]]:
    """
    Parse constraints from a natural language query.
    
    Args:
        query: Natural language query string
        
    Returns:
        Dictionary with keys 'colors', 'garments', 'contexts' containing sets of constraints
    """
    # Same extraction logic as for captions
    return extract_tags(query)


def compute_constraint_score(
    query_constraints: Dict[str, Set[str]],
    item_tags: Dict[str, Set[str]]
) -> float:
    """
    Compute constraint satisfaction score.
    
    Args:
        query_constraints: Constraints extracted from query
        item_tags: Tags extracted from item caption
        
    Returns:
        Score between 0 and 1, where 1 means all constraints are satisfied
    """
    total_constraints = 0
    satisfied_constraints = 0
    
    for category in ['colors', 'garments', 'contexts']:
        query_set = query_constraints.get(category, set())
        item_set = item_tags.get(category, set())
        
        if query_set:
            total_constraints += len(query_set)
            # Count how many query constraints are present in the item
            satisfied = len(query_set & item_set)
            satisfied_constraints += satisfied
    
    if total_constraints == 0:
        # No constraints in query, return 1.0 (fully satisfied by default)
        return 1.0
    
    return satisfied_constraints / total_constraints


def extract_atomic_probes(query: str) -> List[str]:
    """
    Extract atomic attribute probes from query for fine-grained reranking.
    
    Examples:
        "red tie and white shirt" -> ["red tie", "white shirt"]
        "bright yellow raincoat" -> ["bright yellow raincoat", "yellow"]
        "blue shirt on park bench" -> ["blue shirt", "park bench"]
    
    Args:
        query: Natural language query
        
    Returns:
        List of atomic probes (noun phrases with attributes)
    """
    probes = []
    query_lower = query.lower()
    
    # Common color+garment combinations
    for color in COLOR_VOCAB:
        for garment in GARMENT_VOCAB:
            phrase = f"{color} {garment}"
            if phrase in query_lower:
                probes.append(phrase)
    
    # Common color+context combinations
    for color in COLOR_VOCAB:
        for context in CONTEXT_VOCAB:
            phrase = f"{color} {context}"
            if phrase in query_lower:
                probes.append(phrase)
    
    # Standalone colors with emphasis (bright/dark/light)
    for intensity in ['bright', 'dark', 'light', 'pale']:
        for color in COLOR_VOCAB:
            phrase = f"{intensity} {color}"
            if phrase in query_lower:
                probes.append(phrase)
    
    # Context phrases (multi-word)
    context_phrases = [
        'modern office', 'formal setting', 'park bench', 'city walk',
        'beach outfit', 'business attire', 'casual outfit'
    ]
    for phrase in context_phrases:
        if phrase in query_lower:
            probes.append(phrase)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_probes = []
    for p in probes:
        if p not in seen:
            unique_probes.append(p)
            seen.add(p)
    
    # If no probes extracted, use single-word colors and garments
    if not unique_probes:
        for color in COLOR_VOCAB:
            if color in query_lower:
                unique_probes.append(color)
        for garment in GARMENT_VOCAB:
            if garment in query_lower:
                unique_probes.append(garment)
    
    return unique_probes[:5]  # Limit to top 5 probes
