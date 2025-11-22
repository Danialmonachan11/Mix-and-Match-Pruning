"""
LeViT Layer Classification and Sensitivity Scoring (FIXED VERSION)
Implements architecture-aware pruning strategies for LeViT models

KEY FIXES:
1. Head protection (0-30%) instead of aggressive pruning (90-98%)
2. Attention QKV protection (0-30%)
3. MLP protection (0-40%)
4. Aggressive pruning ONLY on projection layers (50-80%)
"""

import re
from typing import Dict, List, Tuple


def classify_levit_layer(layer_name: str, param_count: int) -> str:
    """
    Classify LeViT layer by its architectural role.
    """
    
    # Head layers
    if 'head' in layer_name.lower():
        return 'head'
        
    # Patch embedding
    if 'patch_embed' in layer_name.lower():
        return 'patch_embed'
        
    # Attention layers
    if 'attn' in layer_name.lower():
        if 'qkv' in layer_name.lower():
            return 'attn_qkv'
        elif 'proj' in layer_name.lower():
            return 'attn_proj'
        else:
            return 'attn_other'
            
    # MLP layers
    if 'mlp' in layer_name.lower():
        return 'mlp'
        
    # Normalization
    if 'norm' in layer_name.lower() or 'bn' in layer_name.lower():
        return 'norm'
        
    return 'other'


def extract_levit_stage(layer_name: str) -> int:
    """
    Extract stage number from LeViT layer name.
    LeViT has 3 main stages in its backbone.
    """
    # Try to find stage number in blocks
    match = re.search(r'blocks\.(\d+)', layer_name)
    if match:
        block_idx = int(match.group(1))
        # Map block index to stage (approximate for LeViT-384)
        if block_idx < 4:
            return 1
        elif block_idx < 8:
            return 2
        else:
            return 3
            
    return 0


def compute_levit_sparsity_ranges(
    layer_names: List[str],
    layer_params: Dict[str, int]
) -> Dict[str, Tuple[float, float]]:
    """
    Compute LeViT-specific pruning ranges based on architectural insights.
    
    CORRECTED Strategy (based on sensitivity analysis):
    1. Protect Head (0-30%) - MOST sensitive
    2. Protect Attention QKV (0-30%) - Critical for feature extraction
    3. Protect MLP layers (0-40%) - High sensitivity
    4. Protect Patch Embed (0-20%) - Input processing
    5. Protect Normalization (0-0%) - Stability
    6. Prune Attention Projections (50-80%) - Redundant linear combinations
    """

    layer_ranges = {}

    for layer_name in layer_names:
        param_count = layer_params.get(layer_name, 0)
        layer_role = classify_levit_layer(layer_name, param_count)
        stage = extract_levit_stage(layer_name)

        # RULE 1: Normalization protection
        if layer_role == 'norm':
            layer_ranges[layer_name] = (0.0, 0.0)
            continue

        # RULE 2: Head protection (Critical!)
        if layer_role == 'head':
            layer_ranges[layer_name] = (0.0, 30.0)
            continue

        # RULE 3: Patch Embedding protection
        if layer_role == 'patch_embed':
            layer_ranges[layer_name] = (0.0, 20.0)
            continue

        # RULE 4: Attention QKV protection
        if layer_role == 'attn_qkv':
            layer_ranges[layer_name] = (0.0, 30.0)
            continue

        # RULE 5: MLP protection
        if layer_role == 'mlp':
            layer_ranges[layer_name] = (0.0, 40.0)
            continue

        # RULE 6: Attention Projection (Aggressive Pruning Candidate)
        if layer_role == 'attn_proj':
            layer_ranges[layer_name] = (50.0, 80.0)
            continue

        # Fallback
        layer_ranges[layer_name] = (0.0, 10.0)

    return layer_ranges


def get_levit_sensitivity_ordering(layer_names: List[str]) -> List[str]:
    """
    Order LeViT layers by pruning priority based on sensitivity.
    """
    def get_priority_score(layer_name: str) -> int:
        param_count = 0
        layer_role = classify_levit_layer(layer_name, param_count)
        
        if layer_role == 'attn_proj':
            return 0  # Prune first
        elif layer_role == 'attn_other':
            return 1
        elif layer_role == 'mlp':
            return 2
        elif layer_role == 'attn_qkv':
            return 3
        elif layer_role == 'head':
            return 4
        elif layer_role == 'patch_embed':
            return 5
        elif layer_role == 'norm':
            return 6
        else:
            return 7

    ordered_layers = sorted(layer_names, key=get_priority_score)
    return ordered_layers


def print_levit_layer_analysis(layer_names: List[str], layer_params: Dict[str, int]):
    """Print detailed analysis of LeViT layer classifications and ranges"""
    
    print("\n" + "=" * 80)
    print("LeViT Layer Classification Analysis")
    print("=" * 80)
    
    layer_ranges = compute_levit_sparsity_ranges(layer_names, layer_params)
    
    # Group by role
    by_role = {}
    for layer_name in layer_names:
        param_count = layer_params.get(layer_name, 0)
        role = classify_levit_layer(layer_name, param_count)
        
        if role not in by_role:
            by_role[role] = []
        by_role[role].append((layer_name, param_count, layer_ranges.get(layer_name, (0, 0))))
    
    # Print by role
    for role in sorted(by_role.keys()):
        print(f"\n{role}:")
        for layer_name, param_count, (min_r, max_r) in by_role[role]:
            print(f"  {layer_name:.<45} {param_count:>10,} params | Range: [{min_r:>5.1f}%, {max_r:>5.1f}%]")
    
    print("\n" + "=" * 80)
    print("SUMMARY OF KEY FIXES:")
    print("=" * 80)
    print("1. ✅ Head: 0-30% (Protected)")
    print("2. ✅ Attn QKV: 0-30% (Protected)")
    print("3. ✅ MLP: 0-40% (Protected)")
    print("4. ✅ Attn Proj: 50-80% (Aggressive)")
    print("=" * 80)
