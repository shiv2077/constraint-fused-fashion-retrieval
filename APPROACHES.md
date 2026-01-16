# Approaches to Multimodal Fashion & Context Retrieval

## Problem Statement
Build an intelligent search engine that retrieves fashion images from natural language descriptions, understanding not just "what" (clothing attributes) but "where" (environment/context) and the "vibe" (style).

---

## Approach 1: Vanilla CLIP (Baseline)

### Description
Use CLIP's pre-trained image and text encoders to embed both images and queries into a shared space, then retrieve via cosine similarity.

### Implementation
```python
# Indexing
image_features = clip.encode_image(images)

# Retrieval
text_features = clip.encode_text(query)
similarities = cosine_similarity(text_features, image_features)
```

### Strengths ✅
- Simple, one-model solution
- Strong zero-shot capability
- Fast inference
- Pre-trained on 400M image-text pairs

### Weaknesses ❌
- **Poor compositionality**: Struggles with "red shirt + blue pants" vs "blue shirt + red pants"
- **Weak fine-grained attributes**: Misses specific colors, garment types
- **No explicit constraint enforcement**: Can't ensure all query requirements are met
- **Bag-of-words bias**: Treats "red" and "blue" independently

### When to Use
- Quick prototyping
- General-purpose retrieval
- When speed > accuracy

### Trade-offs
- **Speed**: Fast ⚡ (single forward pass)
- **Accuracy**: Medium 📊
- **Complexity**: Low 🔧
- **Compositionality**: Poor ❌

---

## Approach 2: Fine-tuned Fashion CLIP

### Description
Fine-tune CLIP on fashion-specific dataset (e.g., Fashion200K, FashionIQ) with triplet loss or contrastive learning.

### Implementation
```python
# Fine-tuning
loss = triplet_loss(anchor, positive, negative)

# Retrieval (same as vanilla CLIP)
similarities = cosine_similarity(query_emb, image_embs)
```

### Strengths ✅
- Better fashion-specific understanding
- Improved attribute recognition
- Still maintains zero-shot capability (if done right)

### Weaknesses ❌
- **Requires labeled fashion data** and GPU compute for training
- **Still has compositionality issues** (architectural limitation)
- **Risk of overfitting** to training distribution
- **Expensive to maintain** (needs retraining for new attributes)

### When to Use
- Have labeled fashion data
- Target specific fashion domain
- Can afford training costs

### Trade-offs
- **Speed**: Fast ⚡ (single forward pass)
- **Accuracy**: Medium-High 📈
- **Complexity**: Medium 🔧🔧
- **Compositionality**: Slightly better but still limited

---

## Approach 3: Two-Stage Retrieval (Retrieve + Rerank)

### Description
Use fast vector search for candidate retrieval, then rerank with a more powerful cross-encoder model.

### Implementation
```python
# Stage 1: Fast retrieval
candidates = faiss.search(query_embedding, topN=100)

# Stage 2: Cross-encoder reranking
for img, query in zip(candidate_images, [query]*len(candidates)):
    score = cross_encoder(img, query)  # Sees both together
```

### Strengths ✅
- **Better compositionality**: Cross-encoder sees full image-text pair
- **Higher accuracy**: Two complementary signals
- **Scalable**: Fast first stage + accurate second stage

### Weaknesses ❌
- **Slower**: Two forward passes required
- **More complex**: Need to manage two models
- **topN parameter tuning**: Balance speed vs recall

### When to Use
- Accuracy is critical
- Can tolerate 2-3x latency increase
- Have GPU resources for reranking

### Trade-offs
- **Speed**: Medium ⚡⚡ (two stages)
- **Accuracy**: High 📈📈
- **Complexity**: Medium-High 🔧🔧🔧
- **Compositionality**: Good ✅

---

## Approach 4: Hybrid Multi-Signal Fusion (CHOSEN ⭐)

### Description
Combine three complementary signals:
1. **Dense retrieval** (SigLIP embeddings) - fast semantic matching
2. **Cross-modal reranking** (BLIP ITM) - accurate image-text alignment
3. **Constraint satisfaction** (tag extraction) - explicit attribute matching

### Implementation
```python
# Signal 1: Vector similarity
vec_score = siglip.similarity(query, image)

# Signal 2: Image-text matching
itm_score = blip_itm.score(image, query)

# Signal 3: Constraint satisfaction
query_constraints = parse_constraints(query)  # {colors, garments, contexts}
item_tags = extract_tags(caption)
cons_score = compute_overlap(query_constraints, item_tags)

# Fusion with penalty
penalty = 0.2 if cons_score < 0.5 and has_constraints else 1.0
final = penalty * (0.40*vec + 0.45*itm + 0.15*cons)
```

### Architecture Diagram
```
Query → [SigLIP Text Encoder] → Vector Search (FAISS) → Top-N Candidates
                                                               ↓
Image → [SigLIP Image Encoder] → Vectors                      ↓
Image → [BLIP Caption] → Tags Extraction                      ↓
                                                               ↓
                                    [Multi-Signal Fusion] ← BLIP ITM Score
                                            ↓
                                    Ranked Results
```

### Strengths ✅
- **Strong compositionality**: ITM cross-encoder + explicit constraints
- **Fine-grained attributes**: Tag extraction catches specific details
- **Robust**: Three complementary signals reduce failure modes
- **Explainable**: Can show which signal contributed to ranking
- **Configurable**: Weights can be tuned per use case

### Weaknesses ❌
- **Complex implementation**: Three models to manage
- **Slower**: Three forward passes (SigLIP, BLIP Caption, BLIP ITM)
- **Memory intensive**: Multiple models in GPU memory
- **Requires vocabulary**: Tag extraction needs pre-defined fashion terms

### When to Use
- Fashion-specific retrieval
- Need high accuracy on complex queries
- Want explainability
- Have GPU resources

### Trade-offs
- **Speed**: Slower ⚡⚡⚡ (three models)
- **Accuracy**: Very High 📈📈📈
- **Complexity**: High 🔧🔧🔧🔧
- **Compositionality**: Excellent ✅✅

---

## Approach 5: Vision-Language Models with Attention (e.g., BLIP-2, Flamingo)

### Description
Use large-scale vision-language models that can generate detailed descriptions and answer questions about images.

### Implementation
```python
# Generative approach
description = vl_model.generate(image, "Describe this outfit")
match_score = semantic_similarity(description, query)

# Or discriminative
score = vl_model.score(image, query)
```

### Strengths ✅
- **State-of-the-art accuracy**: Latest architectures
- **Rich understanding**: Can reason about complex scenes
- **Few-shot learning**: Can adapt with few examples

### Weaknesses ❌
- **Very slow**: Large models (10B+ parameters)
- **High memory**: Requires 40GB+ GPU RAM
- **Not scalable**: Can't index millions of images efficiently
- **Overkill**: Too powerful for pure retrieval

### When to Use
- Research/exploration
- Small dataset (<10K images)
- Need detailed image understanding

### Trade-offs
- **Speed**: Very slow 🐌
- **Accuracy**: Very High 📈📈📈
- **Complexity**: Very High 🔧🔧🔧🔧🔧
- **Compositionality**: Excellent ✅✅

---

## Comparison Matrix

| Approach | Speed | Accuracy | Compositionality | Scalability | Complexity | Best For |
|----------|-------|----------|------------------|-------------|------------|----------|
| **Vanilla CLIP** | ⚡⚡⚡⚡⚡ | 📊📊 | ❌ | ⚙️⚙️⚙️⚙️⚙️ | 🔧 | Quick prototypes |
| **Fine-tuned CLIP** | ⚡⚡⚡⚡ | 📊📊📊 | ❌ | ⚙️⚙️⚙️⚙️ | 🔧🔧 | Domain-specific |
| **Retrieve + Rerank** | ⚡⚡⚡ | 📊📊📊📊 | ✅ | ⚙️⚙️⚙️⚙️ | 🔧🔧🔧 | Accuracy-focused |
| **Multi-Signal (Ours)** | ⚡⚡ | 📊📊📊📊📊 | ✅✅ | ⚙️⚙️⚙️ | 🔧🔧🔧🔧 | Fashion retrieval |
| **Large VLMs** | ⚡ | 📊📊📊📊📊 | ✅✅ | ⚙️ | 🔧🔧🔧🔧🔧 | Research |

---

## Why We Chose Approach 4 (Multi-Signal Fusion)

### 1. Addresses Compositionality
The assignment explicitly states vanilla CLIP struggles with compositionality. Our approach addresses this through:
- **BLIP ITM cross-encoder**: Sees full image-text pair together, not separately
- **Explicit constraints**: Tag-based matching ensures "red + shirt" both present

### 2. Fashion-Specific
- **Tag extraction**: Targets fashion attributes (colors, garments, styles)
- **BLIP captions**: Generate rich descriptions for tag extraction
- **Constraint penalty**: Enforces query requirements

### 3. Explainable
Each signal contributes to final score:
```
Result: 0.85 = 0.4×(0.92 vec) + 0.45×(0.88 itm) + 0.15×(0.67 cons)
```
Can show why an image ranked high or low.

### 4. Configurable
Weights can be tuned for different use cases:
- More vector weight → faster, more semantic
- More ITM weight → slower, more accurate
- More constraint weight → stricter attribute matching

### 5. Balanced Trade-offs
- Not as slow as large VLMs
- More accurate than single-model approaches
- Scalable to millions of images (with FAISS IVF)

---

## Model Selection Justification

### SigLIP vs CLIP
- **SigLIP** uses sigmoid loss instead of softmax
- Better calibrated similarities
- Improved zero-shot performance
- Same architecture, better training

### BLIP vs Other Captioners
- **BLIP** specifically trained for image captioning
- High-quality descriptions
- Fast inference
- Widely adopted in research

### BLIP-ITM vs CLIP for Reranking
- **BLIP-ITM** is a cross-encoder (sees both inputs together)
- **CLIP** is a bi-encoder (encodes separately)
- Cross-encoders achieve higher accuracy
- Worth the speed trade-off for top-K reranking

---

## Implementation Details

### Why FAISS?
- **Industry standard** for similarity search
- **GPU acceleration** available
- **Scalable**: Can handle billions of vectors
- **Multiple index types**: Flat, IVF, HNSW

### Why Rule-Based Tag Extraction?
- **Fast**: No model inference needed
- **Explainable**: Clear what was matched
- **Extensible**: Easy to add new terms
- **Robust**: Doesn't fail silently like ML models

### Why Penalty Function?
```python
penalty = 0.2 if constraint_score < 0.5 and has_constraints else 1.0
```
- **Soft constraint**: Doesn't eliminate results, just penalizes
- **Only when needed**: No penalty if query has no constraints
- **Threshold-based**: Clear cutoff at 50% satisfaction
- **Tunable**: Can adjust threshold and penalty factor

---

## What We Didn't Do (and Why)

### ❌ Fine-tuning Models
**Why not**: 
- Assignment emphasizes zero-shot capability
- Pre-trained models already strong
- Requires labeled fashion data + compute

### ❌ Complex Fusion (e.g., learned weights)
**Why not**:
- Simple weighted sum is interpretable
- Works well in practice
- No training data needed

### ❌ Custom Vector Index
**Why not**:
- FAISS is production-grade
- Focus on ML logic, not engineering (per assignment)

### ❌ Multiple Datasets
**Why not**:
- Fashionpedia provides 3200 diverse fashion images
- Exceeds minimum requirement (500-1000)
- Single source ensures consistency

---

## Conclusion

Our **hybrid multi-signal fusion** approach strikes the best balance between:
- **Accuracy** (3 complementary signals)
- **Speed** (FAISS + batched reranking)
- **Compositionality** (cross-encoder + explicit constraints)
- **Scalability** (vector index for first stage)
- **Explainability** (interpretable score breakdown)

It directly addresses the assignment's critique of vanilla CLIP while remaining practical and extensible.
