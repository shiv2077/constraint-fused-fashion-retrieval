import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.retriever.fashion_retriever import FashionRetriever

def test_edge_cases(retriever):
    test_cases = [
        ("", "empty query"),
        ("   ", "whitespace only"),
        ("aslkdjfalksjdflaksjdf", "nonsense query"),
        ("x" * 500, "extremely long query"),
        ("person wearing purple polka dot jacket with neon green stripes", "non-existent attributes"),
        ("red red red red red red red red shirt", "repeated words"),
        ("shirt", "single word"),
        ("!@#$%^&*()", "special characters only"),
        ("wearing", "stop word only"),
        ("red shirt AND blue pants OR yellow jacket", "boolean operators"),
        ("woman in a red dress standing in a park with a dog holding a coffee", "very specific scene"),
        ("elegant", "single abstract adjective"),
        ("123456789", "numbers only"),
        ("RED SHIRT BLUE PANTS", "all caps"),
        ("red shirt blue pants yellow hat green shoes pink bag", "too many colors"),
    ]
    
    results = {}
    errors = []
    
    print("=" * 60)
    print("EDGE CASE TESTING")
    print("=" * 60)
    
    for query, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        
        try:
            res = retriever.search(query, k=5)
            results[description] = {
                'query': query,
                'num_results': len(res) if res else 0,
                'status': 'SUCCESS'
            }
            print(f"✓ Got {len(res) if res else 0} results")
        except Exception as e:
            errors.append({
                'test': description,
                'query': query,
                'error': str(e),
                'type': type(e).__name__
            })
            results[description] = {
                'query': query,
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"✗ FAILED: {type(e).__name__}: {str(e)[:100]}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {len(test_cases) - len(errors)}")
    print(f"Failed: {len(errors)}")
    
    if errors:
        print("\n" + "=" * 60)
        print("FAILURES")
        print("=" * 60)
        for err in errors:
            print(f"\nTest: {err['test']}")
            print(f"Query: {err['query'][:50]}")
            print(f"Error: {err['type']}: {err['error']}")
    
    return results, errors

def test_robustness(index_dir):
    print("=" * 60)
    print("ROBUSTNESS CHECKS")
    print("=" * 60)
    
    checks = []
    
    manifest_path = Path(index_dir) / "manifest.json"
    metadata_path = Path(index_dir) / "metadata.json"
    vectors_path = Path(index_dir) / "vectors.faiss"
    
    if manifest_path.exists():
        import json
        with open(manifest_path) as f:
            manifest = json.load(f)
            print(f"\n✓ Manifest exists")
            print(f"  - Num images: {manifest.get('num_images', 'N/A')}")
            print(f"  - Max images: {manifest.get('max_images', 'N/A')}")
            checks.append(('manifest', True, manifest.get('num_images')))
    else:
        print(f"\n✗ Manifest not found")
        checks.append(('manifest', False, 0))
    
    if metadata_path.exists():
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
            print(f"\n✓ Metadata exists")
            print(f"  - Num entries: {len(metadata)}")
            
            if metadata:
                sample = metadata[0]
                print(f"  - Sample keys: {list(sample.keys())}")
                
                has_caption = 'caption' in sample
                has_colors = 'colors' in sample and sample['colors']
                has_garments = 'garments' in sample and sample['garments']
                has_contexts = 'contexts' in sample and sample['contexts']
                
                print(f"  - Has captions: {'✓' if has_caption else '✗'}")
                print(f"  - Has colors: {'✓' if has_colors else '✗'}")
                print(f"  - Has garments: {'✓' if has_garments else '✗'}")
                print(f"  - Has contexts: {'✓' if has_contexts else '✗'}")
                
                color_count = sum(1 for m in metadata if m.get('colors'))
                garment_count = sum(1 for m in metadata if m.get('garments'))
                context_count = sum(1 for m in metadata if m.get('contexts'))
                
                print(f"  - Color coverage: {color_count}/{len(metadata)} ({100*color_count/len(metadata):.1f}%)")
                print(f"  - Garment coverage: {garment_count}/{len(metadata)} ({100*garment_count/len(metadata):.1f}%)")
                print(f"  - Context coverage: {context_count}/{len(metadata)} ({100*context_count/len(metadata):.1f}%)")
                
            checks.append(('metadata', True, len(metadata)))
    else:
        print(f"\n✗ Metadata not found")
        checks.append(('metadata', False, 0))
    
    if vectors_path.exists():
        size_mb = vectors_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ Vector index exists")
        print(f"  - Size: {size_mb:.2f} MB")
        checks.append(('vectors', True, size_mb))
    else:
        print(f"\n✗ Vector index not found")
        checks.append(('vectors', False, 0))
    
    return checks

if __name__ == "__main__":
    index_dir = "artifacts"
    
    checks = test_robustness(index_dir)
    
    manifest_ok = any(c[0] == 'manifest' and c[1] for c in checks)
    if not manifest_ok:
        print("\n⚠ Index not ready yet. Re-indexing may still be in progress.")
        print("Run this script again after indexing completes.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Loading retriever...")
    print("=" * 60)
    
    try:
        retriever = FashionRetriever(index_dir=index_dir)
        print("✓ Retriever loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load retriever: {e}")
        sys.exit(1)
    
    results, errors = test_edge_cases(retriever)
    
    print("\n" + "=" * 60)
    print("DETERMINISM TEST")
    print("=" * 60)
    print("Running same query 3 times to check consistency...")
    
    test_query = "woman in red dress"
    results_1 = retriever.search(test_query, k=5)
    results_2 = retriever.search(test_query, k=5)
    results_3 = retriever.search(test_query, k=5)
    
    ids_1 = [r['image_id'] for r in results_1] if results_1 else []
    ids_2 = [r['image_id'] for r in results_2] if results_2 else []
    ids_3 = [r['image_id'] for r in results_3] if results_3 else []
    
    if ids_1 == ids_2 == ids_3:
        print("✓ Results are deterministic")
    else:
        print("✗ Results are NOT deterministic!")
        print(f"  Run 1: {ids_1}")
        print(f"  Run 2: {ids_2}")
        print(f"  Run 3: {ids_3}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
    
    if errors:
        print(f"\n⚠ {len(errors)} test(s) failed - review failures above")
        sys.exit(1)
    else:
        print(f"\n✓ All {len(results)} tests passed!")
        sys.exit(0)
