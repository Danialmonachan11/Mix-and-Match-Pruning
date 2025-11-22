"""
ResNet Layer Classification and Sensitivity Scoring (FIXED VERSION)
Implements architecture-aware pruning strategies for ResNet models

KEY FIXES:
1. FC layer protection (0-30%) instead of aggressive pruning (90-98%)
2. Inverted layer4 pruning - now aggressive (75-95%) since it's LEAST sensitive
3. Correct interpretation of sensitivity scores
"""

import re
from typing import Dict, List, Tuple


def classify_resnet_layer(layer_name: str, param_count: int) -> str:
    """
    Classify ResNet layer by its architectural role.

    Returns one of:
    - 'initial_conv': First conv layer (conv1)
    - 'initial_bn': First batchnorm (bn1)
    - 'downsample_conv': Downsampling convolution (critical for skip connections)
    - 'downsample_bn': Downsampling batchnorm
    - 'block_conv': Regular residual block convolution
    - 'block_bn': Regular residual block batchnorm
    - 'fc': Final fully connected layer
    """

    # BatchNorm detection
    is_batchnorm = (
        'bn' in layer_name.lower() or
        'batchnorm' in layer_name.lower() or
        'norm' in layer_name.lower() or
        (param_count <= 2048 and param_count in [64, 128, 256, 512, 1024, 2048])
    )

    # Initial layers
    if layer_name == 'conv1.weight':
        return 'initial_conv'
    if layer_name == 'bn1.weight':
        return 'initial_bn'

    # Final FC layer
    if 'fc' in layer_name.lower() or 'linear' in layer_name.lower():
        return 'fc'

    # Downsampling layers (critical for skip connections)
    if 'downsample' in layer_name.lower():
        # downsample.0 is conv, downsample.1 is bn
        if 'downsample.1' in layer_name or is_batchnorm:
            return 'downsample_bn'
        else:
            return 'downsample_conv'

    # Regular block layers
    if is_batchnorm:
        return 'block_bn'
    else:
        return 'block_conv'


def extract_resnet_stage(layer_name: str) -> int:
    """
    Extract which stage (layer1, layer2, layer3, layer4) a layer belongs to.
    Returns 0 for initial layers, 1-4 for stages, 5 for fc.
    """
    if layer_name in ['conv1.weight', 'bn1.weight']:
        return 0
    if 'fc' in layer_name.lower():
        return 5

    # Extract layer number from pattern like "layer2.0.conv1.weight"
    match = re.match(r'layer(\d+)\.', layer_name)
    if match:
        return int(match.group(1))

    return 0


def extract_block_index(layer_name: str) -> int:
    """Extract block index from layer name (e.g., layer2.0.conv1 -> 0)"""
    match = re.match(r'layer\d+\.(\d+)\.', layer_name)

    if match:
        return int(match.group(1))
    return 0


def compute_resnet_sparsity_ranges(
    layer_names: List[str],
    layer_params: Dict[str, int]
) -> Dict[str, Tuple[float, float]]:
    """
    Compute ResNet-specific pruning ranges based on architectural insights.
    
    CORRECTED Strategy (based on sensitivity analysis):
    1. Protect FC layer (0-30%) - MOST sensitive (1.089e-03)
    2. Protect downsampling layers (0-20%) - critical for skip connections
    3. Protect initial conv (0-30%) - preserves low-level features
    4. Protect all BatchNorm (0-0%) - stabilizes training
    5. Conservative on layer1 (30-50%) - early features, more sensitive
    6. Moderate on layer2 (40-60%)
    7. Moderate-Aggressive on layer3 (50-70%)
    8. Aggressive on layer4 (75-95%) - LEAST sensitive, high-level features
    
    NOTE: This INVERTS the previous logic which incorrectly pruned FC heavily!
    """

    layer_ranges = {}

    for layer_name in layer_names:
        param_count = layer_params.get(layer_name, 0)
        layer_role = classify_resnet_layer(layer_name, param_count)
        stage = extract_resnet_stage(layer_name)
        block_idx = extract_block_index(layer_name)

        # RULE 1: BatchNorm protection (all stages)
        if layer_role in ['initial_bn', 'downsample_bn', 'block_bn']:
            layer_ranges[layer_name] = (0.0, 0.0)
            continue

        # RULE 2: Downsampling convolution protection (critical for skip connections)
        if layer_role == 'downsample_conv':
            layer_ranges[layer_name] = (0.0, 20.0)
            continue

        # RULE 3: Initial convolution (preserves low-level features)
        if layer_role == 'initial_conv':
            layer_ranges[layer_name] = (0.0, 30.0)
            continue

        # RULE 4: Final FC layer (PROTECT - most sensitive!)
        # FIXED: Was (90.0, 98.0) which destroyed the classifier!
        if layer_role == 'fc':
            layer_ranges[layer_name] = (0.0, 30.0)  # ✅ CONSERVATIVE!
            continue

        # RULE 5: Block convolutions - CORRECTED BASED ON SENSITIVITY
        # Lower sensitivity = can prune more aggressively
        # Sensitivity: layer4 < layer3 < layer2 < layer1
        if layer_role == 'block_conv':
            if stage == 1:  # layer1 (most sensitive among blocks)
                layer_ranges[layer_name] = (30.0, 50.0)
            elif stage == 2:  # layer2
                layer_ranges[layer_name] = (40.0, 60.0)
            elif stage == 3:  # layer3
                layer_ranges[layer_name] = (50.0, 70.0)
            elif stage == 4:  # layer4 (LEAST sensitive - prune aggressively!)
                # FIXED: Was (70.0, 85.0), now much more aggressive
                layer_ranges[layer_name] = (75.0, 95.0)  # ✅ AGGRESSIVE!
            else:
                layer_ranges[layer_name] = (40.0, 70.0)
            continue

        # Fallback
        layer_ranges[layer_name] = (0.0, 10.0)

    return layer_ranges


def get_resnet_sensitivity_ordering(layer_names: List[str]) -> List[str]:
    """
    Order ResNet layers by pruning priority based on sensitivity.
    
    CORRECTED Priority (low to high sensitivity = high to low pruning):
    1. Layer4 blocks (LEAST sensitive - prune first)
    2. Layer3 blocks
    3. Layer2 blocks
    4. Layer1 blocks
    5. Downsampling layers (critical)
    6. Initial conv (preserves low-level features)
    7. FC layer (MOST sensitive - protect!)
    8. BatchNorm (protected, last)
    """

    def get_priority_score(layer_name: str) -> Tuple[int, int, int]:
        """
        Returns (priority_class, stage, block_index) for sorting.
        Lower priority_class = prune first (less sensitive)
        """
        param_count = 0  # Placeholder
        layer_role = classify_resnet_layer(layer_name, param_count)
        stage = extract_resnet_stage(layer_name)
        block_idx = extract_block_index(layer_name)

        # CORRECTED Priority classes (lower = prune first = less sensitive)
        if layer_role == 'block_conv' and stage == 4:
            priority_class = 0  # Prune first (least sensitive)
        elif layer_role == 'block_conv' and stage == 3:
            priority_class = 1
        elif layer_role == 'block_conv' and stage == 2:
            priority_class = 2
        elif layer_role == 'block_conv' and stage == 1:
            priority_class = 3
        elif layer_role == 'downsample_conv':
            priority_class = 4
        elif layer_role == 'initial_conv':
            priority_class = 5
        elif layer_role == 'fc':
            priority_class = 6  # FIXED: Was 0, now protected!
        elif 'bn' in layer_role:
            priority_class = 7  # Prune last (protected)
        else:
            priority_class = 8

        # Within same priority class, order by stage (higher stage first for blocks)
        return (priority_class, -stage if priority_class < 4 else stage, block_idx)

    # Sort by priority
    ordered_layers = sorted(layer_names, key=get_priority_score)
    return ordered_layers


def print_resnet_layer_analysis(layer_names: List[str], layer_params: Dict[str, int]):
    """Print detailed analysis of ResNet layer classifications and ranges"""
    
    print("\n" + "=" * 80)
    print("ResNet Layer Classification Analysis")
    print("=" * 80)
    
    layer_ranges = compute_resnet_sparsity_ranges(layer_names, layer_params)
    
    # Group by role
    by_role = {}
    for layer_name in layer_names:
        param_count = layer_params.get(layer_name, 0)
        role = classify_resnet_layer(layer_name, param_count)
        stage = extract_resnet_stage(layer_name)
        
        key = f"{role}_stage{stage}"
        if key not in by_role:
            by_role[key] = []
        by_role[key].append((layer_name, param_count, layer_ranges.get(layer_name, (0, 0))))
    
    # Print by role
    for key in sorted(by_role.keys()):
        print(f"\n{key}:")
        for layer_name, param_count, (min_r, max_r) in by_role[key]:
            print(f"  {layer_name:.<45} {param_count:>10,} params | Range: [{min_r:>5.1f}%, {max_r:>5.1f}%]")
    
    print("\n" + "=" * 80)
    print("SUMMARY OF KEY FIXES:")
    print("=" * 80)
    print("1. ✅ FC layer: 0-30% (was 90-98%) - PROTECTED because MOST sensitive")
    print("2. ✅ Layer4: 75-95% (was 70-85%) - AGGRESSIVE because LEAST sensitive")
    print("3. ✅ Inverted logic to match sensitivity analysis")
    print("4. ✅ Should now match MAG/SNIP/GraSP performance")
    print("=" * 80)


# For backward compatibility
def print_layer_analysis(*args, **kwargs):
    """Alias for backward compatibility"""
    return print_resnet_layer_analysis(*args, **kwargs)
