"""
Swin Transformer Layer Classification and Sensitivity Scoring

Implements architecture-aware pruning strategies for Swin Transformer models
"""

import re
from typing import Dict, List, Tuple

# ============================================================================
# VERIFICATION: CORRECT CODE LOADED (Stage 3 MLP: fc1=60%, fc2=55%)
# ============================================================================
print("=" * 80)
print("NEW_FIX: swin_layer_classifier.py LOADED")
print("  Stage 3 mlp_fc1 max: 60% (reduced from 75%)")
print("  Stage 3 mlp_fc2 max: 55% (reduced from 70%)")
print("  Expected sparsity diversity: ~32-41% (Target: ~40%)")
print("=" * 80)

def classify_swin_layer(layer_name: str, param_count: int) -> str:
    """
    Classify Swin Transformer layer by its architectural role.
    
    Swin architecture:
    - Patch embedding: Initial patch partition and linear projection
    - Window attention: Self-attention within local windows
    - Shifted window attention: Attention with shifted windows
    - MLP blocks: Feed-forward networks  
    - Patch merging: Downsample feature maps (reduce spatial dims)
    - Normalization: LayerNorm layers
    - Head: Classification layer
    
    Returns one of:
    - 'patch_embed': Initial patch embedding
    - 'patch_merge': Patch merging/downsampling layers
    - 'window_attn_qkv': Window attention Q/K/V projections
    - 'window_attn_proj': Window attention output projection
    - 'mlp_fc1': MLP first linear layer (expansion)
    - 'mlp_fc2': MLP second linear layer (projection)
    - 'norm': Normalization layers
    - 'head': Classification head
    - 'other': Other layers
    """
    layer_lower = layer_name.lower()
    
    # Patch embedding (stem)
    if 'patch_embed' in layer_lower or 'patch_partition' in layer_lower:
        return 'patch_embed'
    
    # Patch merging (downsampling between stages)
    if 'downsample' in layer_lower or 'patch_merge' in layer_lower or 'reduction' in layer_lower:
        return 'patch_merge'
    
    # Window attention Q/K/V
    if 'attn' in layer_lower and 'qkv' in layer_lower:
        return 'window_attn_qkv'
    
    # Window attention projection
    if 'attn' in layer_lower and 'proj' in layer_lower and 'qkv' not in layer_lower:
        return 'window_attn_proj'
    
    # MLP layers
    if 'mlp' in layer_lower:
        if 'fc1' in layer_lower or 'w1' in layer_lower or 'in_proj' in layer_lower:
            return 'mlp_fc1'
        elif 'fc2' in layer_lower or 'w2' in layer_lower or 'out_proj' in layer_lower:
            return 'mlp_fc2'
        else:
            return 'mlp_fc1'  # Default MLP layer
    
    # Normalization
    if 'norm' in layer_lower or 'ln' in layer_lower:
        return 'norm'
    
    # Classification head
    if 'head' in layer_lower or 'fc' in layer_lower:
        # Check if it's the final classification layer (usually large output)
        if param_count > 100000 or 'head' in layer_lower:
            return 'head'
    
    return 'other'


def extract_swin_stage(layer_name: str) -> int:
    """
    Extract stage number from Swin layer name.
    Swin has 4 stages with progressively smaller spatial dimensions.
    
    Returns stage number (0-3) or -1 if not in a stage.
    """
    # Look for patterns like 'layers.0', 'layers.1', 'layers.2', 'layers.3'
    # or 'stage0', 'stage1', etc.
    
    stage_patterns = [
        r'layers\.(\d+)',
        r'layer(\d+)',
        r'stage(\d+)',
        r'blocks\.(\d+)'
    ]
    
    for pattern in stage_patterns:
        match = re.search(pattern, layer_name)
        if match:
            stage_num = int(match.group(1))
            return min(stage_num, 3)  # Cap at stage 3
    
    # Default: early layers (patch embed) = stage 0
    if 'patch_embed' in layer_name.lower():
        return 0
    
    return -1


def compute_swin_sparsity_ranges(
    layer_names: List[str],
    layer_params: Dict[str, int]
) -> Dict[str, Tuple[float, float]]:
    """
    Compute architecture-aware sparsity ranges for Swin Transformer layers.
    
    Pruning philosophy:
    - Patch embedding: Conservative (5-20%) - critical for feature extraction
    - Early stages: More conservative - establish feature hierarchy
    - Later stages: More aggressive - higher-level features more redundant
    - Window attention Q/K/V: Moderate (30-60%) - important but some redundancy
    - Window attention projection: Aggressive (40-70%)
    - MLP layers: Most aggressive (50-80%) - most parameters, highest redundancy
    - Patch merging: Conservative (10-30%) - important for spatial reduction
    - Norms: Very conservative (0-10%) - few params, critical
    - Head: Conservative (5-25%) - final classification
    
    Returns: Dict mapping layer_name -> (min_sparsity%, max_sparsity%)
    """
    ranges = {}
    total_params = sum(layer_params.values())
    
    for layer_name in layer_names:
        param_count = layer_params[layer_name]
        param_ratio = param_count / total_params
        layer_type = classify_swin_layer(layer_name, param_count)
        stage = extract_swin_stage(layer_name)
        
        # Base ranges by layer type - SIGNIFICANTLY REDUCED to hit ~40% target
        if layer_type == 'patch_embed':
            min_r, max_r = 0.0, 10.0  # Was 5-20
        
        elif layer_type == 'patch_merge':
            min_r, max_r = 5.0, 20.0  # Was 10-30
        
        elif layer_type == 'window_attn_qkv':
            # Stage-dependent: more aggressive in later stages
            if stage == 0:
                min_r, max_r = 10.0, 30.0  # Was 25-50
            elif stage == 1:
                min_r, max_r = 15.0, 35.0  # Was 30-55
            elif stage == 2:
                min_r, max_r = 20.0, 40.0  # Was 35-60
            else:  # stage 3
                min_r, max_r = 25.0, 45.0  # Was 40-65

        elif layer_type == 'window_attn_proj':
            # More aggressive than Q/K/V
            if stage == 0:
                min_r, max_r = 20.0, 40.0  # Was 35-60
            elif stage == 1:
                min_r, max_r = 25.0, 45.0  # Was 40-65
            elif stage == 2:
                min_r, max_r = 30.0, 50.0  # Was 45-70
            else:  # stage 3
                min_r, max_r = 35.0, 55.0  # Was 50-75

        elif layer_type == 'mlp_fc1':
            # Most aggressive - expansion layer
            if stage == 0:
                min_r, max_r = 30.0, 45.0  # Was 45-60
            elif stage == 1:
                min_r, max_r = 35.0, 50.0  # Was 50-65
            elif stage == 2:
                min_r, max_r = 40.0, 55.0  # Was 55-70
            else:  # stage 3
                min_r, max_r = 45.0, 60.0  # Was 60-75

        elif layer_type == 'mlp_fc2':
            # Slightly less aggressive than fc1
            if stage == 0:
                min_r, max_r = 25.0, 40.0  # Was 40-55
            elif stage == 1:
                min_r, max_r = 30.0, 45.0  # Was 45-60
            elif stage == 2:
                min_r, max_r = 35.0, 50.0  # Was 50-65
            else:  # stage 3
                min_r, max_r = 40.0, 55.0  # Was 55-70
        
        elif layer_type == 'norm':
            min_r, max_r = 0.0, 5.0  # Very conservative
        
        elif layer_type == 'head':
            min_r, max_r = 0.0, 15.0  # Was 5-25
        
        else:  # 'other'
            min_r, max_r = 10.0, 30.0
        
        ranges[layer_name] = (min_r, max_r)
    
    return ranges


def get_swin_sensitivity_ordering(layer_names: List[str]) -> List[str]:
    """
    Order layers by sensitivity for pruning.
    Most sensitive (prune last) -> Least sensitive (prune first)
    
    Sensitivity hierarchy (most to least sensitive):
    1. Patch embedding (stem) - critical
    2. Patch merging - spatial reduction
    3. Early stage layers - foundation features
    4. Window attention Q/K/V
    5. Window attention projection
    6. MLP fc2
    7. MLP fc1 (least sensitive - most redundant)
    8. Classification head
    """
    
    sensitivity_scores = {}
    
    for layer_name in layer_names:
        layer_type = classify_swin_layer(layer_name, 0)
        stage = extract_swin_stage(layer_name)
        
        # Base sensitivity score (higher = more sensitive = prune later)
        if layer_type == 'patch_embed':
            base_score = 100
        elif layer_type == 'patch_merge':
            base_score = 90
        elif layer_type == 'window_attn_qkv':
            base_score = 70
        elif layer_type == 'window_attn_proj':
            base_score = 60
        elif layer_type == 'mlp_fc2':
            base_score = 40
        elif layer_type == 'mlp_fc1':
            base_score = 30
        elif layer_type == 'head':
            base_score = 80
        elif layer_type == 'norm':
            base_score = 95
        else:
            base_score = 50
        
        # Adjust by stage (earlier stages more sensitive)
        if stage >= 0:
            stage_penalty = stage * 5  # Later stages less sensitive
            base_score -= stage_penalty
        
        sensitivity_scores[layer_name] = base_score
    
    # Sort by sensitivity score (descending - most sensitive first)
    # Then reverse to get least sensitive first for pruning order
    sorted_layers = sorted(layer_names, key=lambda x: sensitivity_scores[x], reverse=False)
    
    return sorted_layers


def print_swin_layer_analysis(layer_names: List[str], layer_params: Dict[str, int]):
    """Print analysis of Swin Transformer layer distribution"""
    
    type_counts = {}
    type_params = {}
    stage_params = {0: 0, 1: 0, 2: 0, 3: 0, -1: 0}
    
    total_params = sum(layer_params.values())
    
    for layer_name in layer_names:
        layer_type = classify_swin_layer(layer_name, layer_params[layer_name])
        stage = extract_swin_stage(layer_name)
        params = layer_params[layer_name]
        
        # Count by type
        type_counts[layer_type] = type_counts.get(layer_type, 0) + 1
        type_params[layer_type] = type_params.get(layer_type, 0) + params
        
        # Count by stage
        if stage >= 0:
            stage_params[stage] += params
        else:
            stage_params[-1] += params
    
    print("\nSwin Transformer Layer Analysis:")
    print("=" * 80)
    print(f"{'Layer Type':<30} {'Count':<10} {'Parameters':<15} {'% of Total':<10}")
    print("-" * 80)
    
    for layer_type in sorted(type_counts.keys()):
        count = type_counts[layer_type]
        params = type_params[layer_type]
        pct = (params / total_params) * 100
        print(f"{layer_type:<30} {count:<10} {params:<15,} {pct:<10.2f}%")
    
    print("\nParameters by Stage:")
    print("-" * 80)
    for stage in sorted([s for s in stage_params.keys() if s >= 0]):
        params = stage_params[stage]
        pct = (params / total_params) * 100 if total_params > 0 else 0
        print(f"  Stage {stage}: {params:,} parameters ({pct:.2f}%)")
    
    if stage_params[-1] > 0:
        params = stage_params[-1]
        pct = (params / total_params) * 100 if total_params > 0 else 0
        print(f"  Other: {params:,} parameters ({pct:.2f}%)")
