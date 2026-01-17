# Testing & Validation Plan

## Current Status
- **Indexing Progress**: 619/3200 (19%)
- **ETA**: ~39 minutes
- **Speed**: 1.11 images/sec

## Test Suite Overview

### 1. Edge Case Testing (`test_edge_cases.py`)
Tests robustness of the system with unusual inputs:

#### Test Cases:
1. **Empty query** - Tests handling of blank input
2. **Whitespace only** - Tests string trimming
3. **Nonsense query** - Tests graceful failure
4. **Extremely long query** (500+ chars) - Tests input length limits
5. **Non-existent attributes** - Purple polka dots with neon green stripes
6. **Repeated words** - "red red red shirt"
7. **Single word** - Minimal input
8. **Special characters** - "!@#$%^&*()"
9. **Stop words only** - "wearing"
10. **Boolean operators** - "AND", "OR" in query
11. **Very specific scene** - Complex multi-object query
12. **Single abstract adjective** - "elegant"
13. **Numbers only** - "123456789"
14. **All caps** - "RED SHIRT BLUE PANTS"
15. **Too many colors** - 5+ colors in one query

#### Expected Behavior:
- No crashes
- Graceful handling of edge cases
- Returns empty list or low-confidence results for nonsense
- Maintains deterministic results

### 2. Robustness Checks (`test_edge_cases.py`)
Validates index integrity:
- ✓ Manifest exists and has correct image count
- ✓ Metadata has 3200 entries
- ✓ All metadata entries have required fields
- ✓ Tag coverage statistics (colors/garments/contexts)
- ✓ Vector index file exists and is reasonable size

### 3. Determinism Test
Runs same query 3 times, validates:
- Same image IDs returned
- Same ranking order
- Same scores

### 4. Full Evaluation Suite (`src/evaluation/run_prompts.py`)
Runs 5 assignment queries on full 3200 images:

1. **"woman wearing a yellow raincoat"**
2. **"modern office setting with glass walls"**
3. **"person in a blue formal shirt"**
4. **"casual streetwear with sneakers"**
5. **"outdoor fashion shoot in natural lighting"**

#### Metrics Computed:
- Precision@5
- Recall@5
- Number of relevant results
- Average metrics across all queries

#### Expected Improvements (vs 25-image index):
- **P@5**: 0.2 → 0.5-0.7
- **R@5**: 0.4 → 0.6-0.8
- More diverse results
- Better color matching
- Better context matching

### 5. Comparison Analysis
Compare old (25 images) vs new (3200 images):
- Metric improvements
- Result diversity
- Tag coverage
- Search quality

## Automation Scripts

### `check_progress.py`
Quick status check with progress bar
```bash
python check_progress.py
```

### `wait_and_test.sh`
Automated test pipeline:
1. Wait for indexing completion
2. Verify index integrity
3. Run edge case tests
4. Run full evaluation
5. Generate metrics report
```bash
./wait_and_test.sh
```

### `monitor_indexing.sh`
Real-time monitoring
```bash
watch -n 5 ./monitor_indexing.sh
```

## Manual Testing Commands

### Check Progress
```bash
python check_progress.py
```

### Run Edge Cases (after indexing)
```bash
python test_edge_cases.py
```

### Run Full Evaluation (after indexing)
```bash
python -m src.evaluation.run_prompts --index_dir artifacts --out_dir evaluation_final
```

### View Results
```bash
cat evaluation_final/results.json | python -m json.tool
```

## Success Criteria

### Must Pass:
- ✓ All 3200 images indexed
- ✓ No crashes on edge cases
- ✓ Deterministic results
- ✓ P@5 ≥ 0.5 (avg across queries)
- ✓ R@5 ≥ 0.5 (avg across queries)

### Should Pass:
- Tag coverage > 60% for colors/garments
- At least 3/5 queries have P@5 ≥ 0.6
- No memory errors or GPU OOM
- All queries complete in < 5 seconds

## Known Limitations

1. **Context tags**: Low coverage (~4%) due to runway-focused dataset
2. **Color ambiguity**: System uses 11 discrete colors
3. **Probe extraction**: May miss complex compositional queries
4. **ITM reranking**: Adds ~2 seconds per query

## Next Steps After Tests

1. Review failed edge cases (if any)
2. Analyze low-performing queries
3. Generate final metrics report
4. Create PDF submission document
5. Commit and push final version
