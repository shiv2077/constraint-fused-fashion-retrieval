# Future Work & Extensions

This document outlines potential enhancements to the constraint-fused fashion retrieval system, focusing on the assignment's specific requirements: **location/place**, **weather**, and **improving precision**.

---

## 1. Location & Place Integration

### Problem Statement
Current system has limited location awareness:
- Can match basic contexts ("office", "park", "street")
- Cannot distinguish specific locations ("Times Square", "Central Park", "Eiffel Tower")
- No geographic reasoning ("clothing for tropical climates")

### 1.1 Hierarchical Place Taxonomy

**Approach**: Build a structured place vocabulary with hierarchies.

```python
PLACE_HIERARCHY = {
    "indoor": {
        "office": ["corporate office", "home office", "coworking space"],
        "retail": ["mall", "boutique", "department store"],
        "dining": ["restaurant", "cafe", "bar"],
        "entertainment": ["theater", "cinema", "concert hall"]
    },
    "outdoor": {
        "urban": ["street", "sidewalk", "plaza", "downtown"],
        "nature": ["park", "forest", "beach", "mountain"],
        "transit": ["airport", "train station", "bus stop"]
    }
}
```

**Implementation**:
```python
def extract_place_tags(caption: str) -> dict:
    """Extract hierarchical place tags from caption."""
    places = {"indoor": [], "outdoor": [], "specific": []}
    
    caption_lower = caption.lower()
    
    # Check each category
    for category, subcategories in PLACE_HIERARCHY.items():
        for subcat, keywords in subcategories.items():
            if any(kw in caption_lower for kw in keywords):
                places[category].append(subcat)
    
    return places
```

**Query Enhancement**:
```python
query = "What to wear to a wedding at the beach"
places = extract_place_tags(query)
# {"outdoor": ["nature"], "specific": ["beach"]}
```

**Scoring**:
```python
def compute_place_score(query_places, item_places):
    """Score based on place hierarchy match."""
    # Exact match: +1.0
    # Category match: +0.5
    # No match: 0.0
    
    score = 0.0
    
    # Check for exact matches
    for category in ["indoor", "outdoor"]:
        query_set = set(query_places.get(category, []))
        item_set = set(item_places.get(category, []))
        exact_matches = query_set & item_set
        score += len(exact_matches) * 1.0
        
        # Partial category match
        if query_set and item_set and not exact_matches:
            score += 0.5
    
    return score
```

**Benefits**:
- ✅ Better context understanding
- ✅ Hierarchical reasoning ("outdoor" matches "park", "beach", "mountain")
- ✅ Extensible taxonomy

**Limitations**:
- ❌ Still rule-based
- ❌ Requires manual vocabulary curation
- ❌ Cannot infer implicit locations

---

### 1.2 Geolocation Embeddings

**Approach**: Use location embeddings to capture geographic semantics.

**Model**: S2 Cell embeddings + GeoEncoder
- [S2 Cells](http://s2geometry.io/): Hierarchical spatial indexing
- Train encoder: Location coordinates → embedding

```python
from s2geometry import CellId, LatLng

class GeoEncoder:
    def __init__(self):
        self.model = self._load_pretrained_geo_encoder()
    
    def encode(self, lat: float, lon: float) -> np.ndarray:
        """Encode lat/lon into 128-dim vector."""
        cell_id = CellId.from_lat_lng(LatLng.from_degrees(lat, lon))
        embedding = self.model.encode(cell_id)
        return embedding
```

**Integration**:
```python
# Indexing: Store location if available in metadata
if "latitude" in metadata and "longitude" in metadata:
    loc_embedding = geo_encoder.encode(metadata["latitude"], metadata["longitude"])
    
# Retrieval: Parse location from query
if location_detected_in_query:
    query_loc_embedding = geo_encoder.encode(parsed_lat, parsed_lon)
    loc_score = cosine_similarity(query_loc_embedding, item_loc_embedding)
```

**Query Examples**:
- "Summer outfit for someone in Paris"
- "Business attire for New York winter"
- "Beach wear in Bali"

**Benefits**:
- ✅ Captures geographic proximity
- ✅ Can learn climate associations (tropical vs arctic)
- ✅ Works with any location (not limited to vocabulary)

**Limitations**:
- ❌ Requires location-labeled training data
- ❌ Fashionpedia dataset lacks geolocation metadata
- ❌ Adds complexity and latency

**Recommendation**: Start with hierarchical taxonomy (1.1), add geolocation (1.2) if location-labeled data becomes available.

---

### 1.3 Visual Place Recognition

**Approach**: Detect locations from image backgrounds using visual features.

**Model**: PlacesCNN or CLIP-Places
- [Places365-CNN](http://places2.csail.mit.edu/): Trained on 365 scene categories
- Outputs: {"outdoor": 0.8, "street": 0.6, "office": 0.1, ...}

```python
class VisualPlaceDetector:
    def __init__(self):
        self.places_model = load_places365_model()
    
    def detect_place(self, image: PIL.Image) -> dict:
        """Detect place category from image background."""
        place_probs = self.places_model(image)
        
        # Get top-3 places
        top_places = sorted(place_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {place: prob for place, prob in top_places}
```

**Integration**:
```python
# Indexing
visual_places = place_detector.detect_place(img)
metadata["visual_places"] = visual_places

# Retrieval
if "office" in query:
    # Boost items with high "office" place probability
    if "office" in item_metadata["visual_places"]:
        place_boost = item_metadata["visual_places"]["office"]
        final_score *= (1 + 0.2 * place_boost)
```

**Benefits**:
- ✅ No geolocation metadata required
- ✅ Works on pure visual features
- ✅ Can detect implicit settings (e.g., formal event from background)

**Limitations**:
- ❌ Fashion images often have generic backgrounds (white, runway)
- ❌ Additional model increases memory/latency
- ❌ May detect irrelevant background details

**Recommendation**: Useful if dataset has diverse real-world backgrounds. Skip for studio/runway datasets like Fashionpedia.

---

## 2. Weather Integration

### Problem Statement
Current system cannot reason about weather appropriateness:
- "What to wear in rainy London" → Cannot prioritize raincoats, umbrellas
- "Summer dress for hot weather" → No temperature awareness
- "Warm coat for winter" → Cannot rank by insulation level

### 2.1 Weather-Clothing Attribute Mapping

**Approach**: Define explicit mappings between weather conditions and garment attributes.

```python
WEATHER_CLOTHING_MAP = {
    "rain": {
        "required": ["raincoat", "umbrella", "waterproof", "jacket"],
        "colors": ["yellow", "navy", "black"],  # Typical rain gear colors
        "materials": ["nylon", "polyester", "rubber"]
    },
    "snow": {
        "required": ["coat", "jacket", "boots", "scarf", "gloves"],
        "attributes": ["warm", "insulated", "thermal", "fleece"],
        "avoid": ["sandals", "shorts", "sleeveless"]
    },
    "summer": {
        "preferred": ["dress", "shorts", "tank top", "sandals", "sunglasses"],
        "attributes": ["light", "breathable", "cotton", "linen"],
        "colors": ["white", "light blue", "pastel"],
        "avoid": ["coat", "boots", "heavy"]
    },
    "winter": {
        "required": ["coat", "jacket", "boots", "scarf"],
        "attributes": ["warm", "wool", "thermal", "insulated"],
        "colors": ["dark", "black", "navy", "gray"],
        "avoid": ["sandals", "shorts", "sleeveless"]
    }
}
```

**Query Parsing**:
```python
def parse_weather_query(query: str) -> dict:
    """Extract weather conditions from query."""
    weather = {
        "conditions": [],
        "temperature": None,
        "season": None
    }
    
    query_lower = query.lower()
    
    # Detect weather keywords
    if any(w in query_lower for w in ["rain", "rainy", "wet"]):
        weather["conditions"].append("rain")
    
    if any(w in query_lower for w in ["snow", "snowy", "blizzard"]):
        weather["conditions"].append("snow")
    
    # Detect season
    for season in ["summer", "winter", "spring", "fall", "autumn"]:
        if season in query_lower:
            weather["season"] = season
    
    # Detect temperature
    if any(w in query_lower for w in ["hot", "warm", "heat"]):
        weather["temperature"] = "hot"
    elif any(w in query_lower for w in ["cold", "freezing", "chilly"]):
        weather["temperature"] = "cold"
    
    return weather
```

**Weather-Aware Scoring**:
```python
def compute_weather_score(item_tags, query_weather):
    """Score item based on weather appropriateness."""
    if not query_weather["conditions"] and not query_weather["season"]:
        return 1.0  # No weather constraints
    
    score = 0.0
    total_checks = 0
    
    # Check for required items
    if query_weather["conditions"]:
        for condition in query_weather["conditions"]:
            required = WEATHER_CLOTHING_MAP[condition]["required"]
            has_required = any(r in item_tags["garments"] for r in required)
            score += 1.0 if has_required else 0.0
            total_checks += 1
    
    # Check for appropriate season
    if query_weather["season"]:
        season_prefs = WEATHER_CLOTHING_MAP[query_weather["season"]]["preferred"]
        has_seasonal = any(p in item_tags["garments"] for p in season_prefs)
        score += 1.0 if has_seasonal else 0.5  # Partial credit
        total_checks += 1
    
    # Check for items to avoid
    if query_weather["season"]:
        avoid_items = WEATHER_CLOTHING_MAP[query_weather["season"]].get("avoid", [])
        has_avoid = any(a in item_tags["garments"] for a in avoid_items)
        if has_avoid:
            score -= 0.5  # Penalty
    
    return score / total_checks if total_checks > 0 else 1.0
```

**Integration into Retrieval**:
```python
# Parse weather from query
query_weather = parse_weather_query(query)

# Compute weather score for each candidate
for idx, _ in candidates:
    item_tags = metadata[idx]["tags"]
    weather_score = compute_weather_score(item_tags, query_weather)
    
    # Add to fusion (new weight: w_weather = 0.10)
    final_score = (
        0.35 * vec_score +      # Reduced from 0.40
        0.40 * itm_score +      # Reduced from 0.45
        0.10 * cons_score +     # Reduced from 0.15
        0.10 * weather_score +  # New signal
        0.05 * place_score      # New signal from section 1
    )
```

**Query Examples**:
- "What to wear in rainy weather" → Boosts raincoats, umbrellas
- "Summer beach outfit" → Prefers light dresses, sandals
- "Warm coat for winter" → Prioritizes insulated jackets, boots

**Benefits**:
- ✅ Explicit, interpretable rules
- ✅ No additional models required
- ✅ Easy to extend with new weather conditions

**Limitations**:
- ❌ Rule-based, not learned
- ❌ Cannot adapt to regional preferences (winter in California vs Canada)
- ❌ Requires manual vocabulary maintenance

---

### 2.2 Weather-Clothing Learned Model

**Approach**: Train a small MLP to predict weather-clothing compatibility.

**Training Data**:
```python
# Pseudo-labeled data from captions
data = [
    {"garment": "raincoat", "weather": "rain", "label": 1.0},
    {"garment": "sandals", "weather": "snow", "label": 0.0},
    {"garment": "coat", "weather": "winter", "label": 1.0},
    ...
]
```

**Model**:
```python
class WeatherClothingMatcher(nn.Module):
    def __init__(self, garment_vocab_size, weather_vocab_size):
        super().__init__()
        self.garment_emb = nn.Embedding(garment_vocab_size, 64)
        self.weather_emb = nn.Embedding(weather_vocab_size, 64)
        
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, garment_ids, weather_ids):
        garment_vec = self.garment_emb(garment_ids)
        weather_vec = self.weather_emb(weather_ids)
        combined = torch.cat([garment_vec, weather_vec], dim=-1)
        return self.mlp(combined)
```

**Usage**:
```python
# At retrieval time
weather_id = weather_vocab.get_id(query_weather)
for garment in item_tags["garments"]:
    garment_id = garment_vocab.get_id(garment)
    compatibility = weather_model(garment_id, weather_id)
    weather_score += compatibility

weather_score /= len(item_tags["garments"])  # Average
```

**Benefits**:
- ✅ Learns from data (more flexible than rules)
- ✅ Can capture subtle patterns
- ✅ Generalizes to unseen combinations

**Limitations**:
- ❌ Requires training data
- ❌ Adds model complexity
- ❌ Less interpretable than rules

**Recommendation**: Start with rule-based (2.1), move to learned (2.2) if you have labeled data or user feedback.

---

### 2.3 Real-Time Weather API Integration

**Approach**: Fetch current weather for detected location and adjust rankings.

**Implementation**:
```python
import requests

def get_weather(location: str) -> dict:
    """Fetch current weather from API."""
    api_key = os.getenv("WEATHER_API_KEY")
    response = requests.get(
        f"https://api.openweathermap.org/data/2.5/weather",
        params={"q": location, "appid": api_key}
    )
    weather_data = response.json()
    
    return {
        "temperature": weather_data["main"]["temp"],
        "condition": weather_data["weather"][0]["main"],  # "Rain", "Clear", etc.
        "description": weather_data["weather"][0]["description"]
    }
```

**Query Enhancement**:
```python
query = "What should I wear in New York today?"

# Parse location
location = extract_location(query)  # "New York"

# Fetch weather
weather = get_weather(location)
# {"temperature": 45, "condition": "Rain", "description": "light rain"}

# Augment query
augmented_query = f"{query} (rainy, 45°F)"
```

**Benefits**:
- ✅ Real-time, dynamic recommendations
- ✅ No need for user to specify weather
- ✅ Useful for "outfit for today" queries

**Limitations**:
- ❌ Requires location detection (not always accurate)
- ❌ API costs and rate limits
- ❌ Adds latency (~200ms for API call)

**Recommendation**: Add as optional feature for location-aware queries. Cache results per location to reduce API calls.

---

## 3. Improving Precision

### Problem Statement
Current system can have false positives:
- "Red dress" → Returns "dress with red background"
- "Blue shirt and black pants" → Returns "blue pants and black shirt"
- "Formal suit" → Returns casual outfits with suit jacket

### 3.1 Hard Negative Mining

**Approach**: Actively find and penalize common failure cases.

**Process**:
1. Collect queries where system fails (user feedback or evaluation)
2. Identify retrieved items that *should not* match
3. Use as hard negatives for fine-tuning

**Example**:
```python
# Query: "Red dress"
# Positive: Image of person in red dress
# Hard Negative: Image of person in blue dress on red carpet (false positive)

triplet_loss = max(0, similarity(query, negative) - similarity(query, positive) + margin)
```

**Fine-tuning**:
```python
def fine_tune_embedder(model, triplets, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    for epoch in range(epochs):
        for query, positive, negative in triplets:
            # Forward pass
            q_emb = model.encode_text(query)
            p_emb = model.encode_image(positive)
            n_emb = model.encode_image(negative)
            
            # Triplet loss
            pos_sim = cosine_similarity(q_emb, p_emb)
            neg_sim = cosine_similarity(q_emb, n_emb)
            loss = max(0, neg_sim - pos_sim + 0.2)
            
            # Backward pass
            loss.backward()
            optimizer.step()
```

**Benefits**:
- ✅ Directly addresses observed failures
- ✅ Improves model on edge cases
- ✅ Proven effective in retrieval tasks

**Limitations**:
- ❌ Requires labeled data (positive + negative pairs)
- ❌ Risk of overfitting to hard negatives
- ❌ Computationally expensive

---

### 3.2 Query Expansion & Reformulation

**Approach**: Generate multiple query variations to increase recall and precision.

**Techniques**:

**A. Synonym Expansion**
```python
SYNONYMS = {
    "dress": ["gown", "frock", "outfit"],
    "red": ["crimson", "scarlet", "ruby"],
    "formal": ["business", "professional", "elegant"]
}

def expand_query(query: str) -> list[str]:
    """Generate query variations with synonyms."""
    expansions = [query]
    
    for word, synonyms in SYNONYMS.items():
        if word in query.lower():
            for syn in synonyms:
                expanded = query.lower().replace(word, syn)
                expansions.append(expanded)
    
    return expansions
```

**B. LLM-based Paraphrasing**
```python
def paraphrase_query(query: str, model="gpt-3.5-turbo") -> list[str]:
    """Generate paraphrases using LLM."""
    prompt = f"""Generate 3 paraphrases of this fashion query:
    Query: "{query}"
    
    Paraphrases:
    1."""
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    paraphrases = parse_paraphrases(response.choices[0].message.content)
    return [query] + paraphrases
```

**C. Attribute Decomposition**
```python
def decompose_query(query: str) -> list[str]:
    """Break complex query into sub-queries."""
    # "Red dress with white collar" →
    # ["red dress", "dress with white collar", "red clothing with white details"]
    
    constraints = parse_query_constraints(query)
    
    sub_queries = []
    
    # Color + garment
    for color in constraints["colors"]:
        for garment in constraints["garments"]:
            sub_queries.append(f"{color} {garment}")
    
    # Full query
    sub_queries.append(query)
    
    return sub_queries
```

**Fusion Strategy**:
```python
# Retrieve for each query variant
all_results = []
for q in query_variants:
    results = search_and_rerank(q, topN=20)
    all_results.extend(results)

# Merge by summing scores
merged = {}
for item_id, score in all_results:
    merged[item_id] = merged.get(item_id, 0) + score

# Sort and deduplicate
final_results = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:10]
```

**Benefits**:
- ✅ Increases recall (finds more relevant items)
- ✅ Handles vocabulary mismatch
- ✅ Robust to query phrasing

**Limitations**:
- ❌ Multiple searches increase latency
- ❌ May introduce noise
- ❌ Requires careful fusion strategy

---

### 3.3 Attribute-Specific Validation

**Approach**: Add specialized validators for critical attributes.

**Color Validator**:
```python
import cv2
import numpy as np

def validate_color(image: PIL.Image, expected_color: str) -> float:
    """Check if image actually contains expected color."""
    # Convert to RGB numpy array
    img_np = np.array(image)
    
    # Define color ranges (HSV)
    COLOR_RANGES = {
        "red": [(0, 100, 100), (10, 255, 255)],
        "blue": [(100, 100, 100), (130, 255, 255)],
        "yellow": [(20, 100, 100), (30, 255, 255)],
        # ... more colors
    }
    
    if expected_color not in COLOR_RANGES:
        return 1.0  # Cannot validate
    
    # Convert to HSV
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    
    # Create mask for color range
    lower, upper = COLOR_RANGES[expected_color]
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    
    # Calculate percentage of pixels in color range
    color_ratio = np.sum(mask > 0) / (image.width * image.height)
    
    return color_ratio
```

**Garment Detector**:
```python
from transformers import AutoModelForImageSegmentation

class GarmentSegmenter:
    def __init__(self):
        self.model = AutoModelForImageSegmentation.from_pretrained("fashion-segmenter")
    
    def detect_garments(self, image: PIL.Image) -> dict:
        """Detect specific garments in image."""
        segmentation = self.model(image)
        
        detected = {
            "dress": has_segment(segmentation, "dress"),
            "shirt": has_segment(segmentation, "shirt"),
            "pants": has_segment(segmentation, "pants"),
            # ... more garments
        }
        
        return detected
```

**Integration**:
```python
# After retrieval, validate top results
for idx, score in top_results:
    # Check colors
    if query_constraints["colors"]:
        for color in query_constraints["colors"]:
            color_validity = validate_color(item_image, color)
            if color_validity < 0.1:  # Less than 10% of expected color
                score *= 0.5  # Penalize
    
    # Check garments
    detected_garments = garment_segmenter.detect_garments(item_image)
    for garment in query_constraints["garments"]:
        if garment not in detected_garments or not detected_garments[garment]:
            score *= 0.7  # Penalize
```

**Benefits**:
- ✅ Directly validates visual attributes
- ✅ Reduces false positives
- ✅ Can catch caption errors

**Limitations**:
- ❌ Additional models increase complexity
- ❌ Slower (especially segmentation)
- ❌ Requires per-attribute validators

---

### 3.4 User Feedback Loop

**Approach**: Learn from user interactions to improve ranking.

**Feedback Types**:
1. **Explicit**: User clicks "relevant" or "not relevant"
2. **Implicit**: Click-through rate, dwell time

**Implementation**:
```python
class FeedbackStore:
    def __init__(self):
        self.feedback = defaultdict(list)  # query → [(item_id, relevance), ...]
    
    def add_feedback(self, query: str, item_id: int, relevance: float):
        """Store user feedback."""
        self.feedback[query].append((item_id, relevance))
    
    def get_boost(self, query: str, item_id: int) -> float:
        """Get boost factor based on past feedback."""
        if query not in self.feedback:
            return 1.0
        
        # Calculate average relevance for this item
        item_feedback = [rel for iid, rel in self.feedback[query] if iid == item_id]
        
        if not item_feedback:
            return 1.0
        
        avg_relevance = sum(item_feedback) / len(item_feedback)
        
        # Boost if relevant, penalize if not
        return 0.5 + avg_relevance  # Range: [0.5, 1.5]
```

**Retrieval Integration**:
```python
# Apply feedback boost
for idx, score in candidates:
    feedback_boost = feedback_store.get_boost(query, idx)
    final_score = score * feedback_boost
```

**Learning from Feedback**:
```python
def train_from_feedback(feedback_data, model, epochs=3):
    """Fine-tune model using feedback as labels."""
    for query, item_id, relevance in feedback_data:
        # Get embeddings
        q_emb = model.encode_text(query)
        i_emb = model.encode_image(item_images[item_id])
        
        # Predicted similarity
        pred_sim = cosine_similarity(q_emb, i_emb)
        
        # Loss: MSE between predicted and user-indicated relevance
        loss = (pred_sim - relevance) ** 2
        
        # Update model
        loss.backward()
        optimizer.step()
```

**Benefits**:
- ✅ Continuously improves over time
- ✅ Adapts to user preferences
- ✅ No manual labeling required

**Limitations**:
- ❌ Requires user traffic
- ❌ Cold start problem (new queries have no feedback)
- ❌ Bias towards popular items

---

## 4. Additional Enhancements

### 4.1 Multi-Modal Query Support

**Approach**: Allow users to query with image + text.

**Example**:
```
User uploads: [Image of red dress]
User types: "but in blue"
```

**Implementation**:
```python
def search_with_image_and_text(reference_image, text_modifier):
    # Encode reference image
    img_emb = siglip.embed_image(reference_image)
    
    # Encode text modifier
    text_emb = siglip.embed_text(text_modifier)
    
    # Combine (weighted average)
    query_emb = 0.7 * img_emb + 0.3 * text_emb
    
    # Search
    results = index.search(query_emb, topN)
    return results
```

---

### 4.2 Personalization

**Approach**: Learn user style preferences over time.

**User Profile**:
```python
user_profile = {
    "preferred_colors": ["black", "navy", "gray"],
    "preferred_styles": ["minimalist", "business"],
    "disliked_garments": ["shorts", "sandals"]
}
```

**Ranking Adjustment**:
```python
for idx, score in candidates:
    tags = metadata[idx]["tags"]
    
    # Boost preferred colors
    for color in user_profile["preferred_colors"]:
        if color in tags["colors"]:
            score *= 1.1
    
    # Penalize disliked garments
    for garment in user_profile["disliked_garments"]:
        if garment in tags["garments"]:
            score *= 0.8
```

---

### 4.3 Trend Detection

**Approach**: Identify emerging fashion trends from data.

**Implementation**:
```python
def detect_trending_attributes(past_queries, time_window_days=30):
    """Find attributes with increasing query frequency."""
    recent = filter_by_time(past_queries, time_window_days)
    older = filter_by_time(past_queries, 2 * time_window_days, time_window_days)
    
    recent_attrs = count_attributes(recent)
    older_attrs = count_attributes(older)
    
    trending = []
    for attr, recent_count in recent_attrs.items():
        older_count = older_attrs.get(attr, 0)
        growth_rate = (recent_count - older_count) / max(older_count, 1)
        
        if growth_rate > 0.5:  # 50% growth
            trending.append((attr, growth_rate))
    
    return sorted(trending, key=lambda x: x[1], reverse=True)
```

---

## 5. Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)
1. **Location taxonomy** (Section 1.1) - Extend attribute parser with place hierarchy
2. **Weather rules** (Section 2.1) - Add weather-clothing mapping
3. **Query expansion** (Section 3.2) - Implement synonym and decomposition

### Phase 2: Validation & Feedback (2-4 weeks)
1. **Color validation** (Section 3.3) - Add HSV-based color checking
2. **Feedback system** (Section 3.4) - Build feedback store and boost mechanism
3. **A/B testing** - Compare with and without new features

### Phase 3: Advanced Features (1-2 months)
1. **Geolocation embeddings** (Section 1.2) - If location data available
2. **Weather API** (Section 2.3) - Real-time weather integration
3. **Hard negative mining** (Section 3.1) - If labeled data available
4. **Garment segmentation** (Section 3.3) - For precise validation

### Phase 4: Long-term (3+ months)
1. **Learned weather model** (Section 2.2) - Train on collected feedback
2. **Visual place detection** (Section 1.3) - If diverse backgrounds in data
3. **Personalization** (Section 4.2) - User-specific ranking
4. **Trend detection** (Section 4.3) - Identify emerging fashion trends

---

## 6. Evaluation Metrics

### Precision & Recall
```python
# For labeled test set
precision = len(retrieved & relevant) / len(retrieved)
recall = len(retrieved & relevant) / len(relevant)
f1 = 2 * precision * recall / (precision + recall)
```

### Mean Average Precision (MAP)
```python
def average_precision(retrieved, relevant):
    """AP for a single query."""
    precisions = []
    num_relevant_seen = 0
    
    for i, item in enumerate(retrieved):
        if item in relevant:
            num_relevant_seen += 1
            precision_at_i = num_relevant_seen / (i + 1)
            precisions.append(precision_at_i)
    
    return sum(precisions) / len(relevant) if relevant else 0.0

map_score = sum(average_precision(r, rel) for r, rel in zip(all_retrieved, all_relevant)) / len(queries)
```

### User Satisfaction
```python
# Click-through rate
ctr = num_clicks / num_queries

# Mean Reciprocal Rank (position of first relevant result)
mrr = sum(1 / rank_of_first_relevant) / num_queries
```

---

## 7. Conclusion

The future work outlined here focuses on three key areas requested in the assignment:

1. **Location/Place**: Hierarchical taxonomy → Visual detection → Geolocation embeddings
2. **Weather**: Rule-based mapping → Learned model → Real-time API
3. **Precision**: Query expansion → Validation → Feedback loop → Hard negatives

**Recommended Priority**:
1. **High**: Location taxonomy, Weather rules, Query expansion (quick wins)
2. **Medium**: Feedback system, Color validation (improve core system)
3. **Low**: Geolocation, Weather API, Segmentation (nice-to-have)

All enhancements are designed to build on the existing three-signal architecture without requiring a complete redesign.
