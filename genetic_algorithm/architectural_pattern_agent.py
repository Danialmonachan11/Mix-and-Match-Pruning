"""Architectural pattern-based pruning agent that reduces 19D space to manageable patterns."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .robust_sensitivity_analyzer import RobustSensitivityAnalyzer


@dataclass 
class ArchitecturalPattern:
    """Defines a pruning pattern based on architectural understanding."""
    pattern_name: str
    description: str
    conv_early_factor: float     # Multiplier for early conv layers (features.0-4)  
    conv_mid_factor: float       # Multiplier for mid conv layers (features.8-15)
    conv_late_factor: float      # Multiplier for late conv layers (features.18-22)
    fc_early_factor: float       # Multiplier for first FC layer (classifier.0)
    fc_mid_factor: float         # Multiplier for middle FC layer (classifier.3)  
    fc_final_factor: float       # Multiplier for final FC layer (classifier.6)


class ArchitecturalPatternAgent:
    """
    Agent that uses architectural patterns instead of 19 independent parameters.
    Reduces search space from 19D to 4D: (base_sparsity, conv_bias, fc_bias, protection_level)
    """
    
    def __init__(self, pattern_params: List[float], model_state_dict: Dict, score_dir_path: str):
        """
        Initialize with 4 pattern parameters instead of 19 layer parameters.
        
        Args:
            pattern_params: [base_sparsity, conv_bias, fc_bias, protection_level]
                - base_sparsity (30-80): Overall target sparsity percentage
                - conv_bias (-20 to +20): Bias toward conv layers (negative = less pruning)  
                - fc_bias (-20 to +20): Bias toward FC layers (negative = less pruning)
                - protection_level (0-100): How much to protect critical layers
        """
        
        if len(pattern_params) != 4:
            raise ValueError(f"Expected 4 pattern parameters, got {len(pattern_params)}")
        
        self.base_sparsity = max(30.0, min(80.0, pattern_params[0]))
        self.conv_bias = max(-20.0, min(20.0, pattern_params[1]))  
        self.fc_bias = max(-20.0, min(20.0, pattern_params[2]))
        self.protection_level = max(0.0, min(100.0, pattern_params[3]))
        
        self.model_state_dict = model_state_dict
        self.score_dir_path = score_dir_path
        
        # Initialize sensitivity analyzer for architectural guidance
        self.sensitivity_analyzer = RobustSensitivityAnalyzer(score_dir_path, model_state_dict)
        
        # Create layer mapping
        self.layer_architecture = self._create_layer_architecture()
        
        # Generate the actual 19 layer parameters from the 4 pattern parameters
        self.generated_layer_params = self._generate_layer_parameters()
        
        print(f"🏗️  Architectural Pattern Agent initialized:")
        print(f"   Base Sparsity: {self.base_sparsity:.1f}%")
        print(f"   Conv Bias: {self.conv_bias:+.1f}% (negative = less conv pruning)")
        print(f"   FC Bias: {self.fc_bias:+.1f}% (negative = less FC pruning)")  
        print(f"   Protection: {self.protection_level:.1f}% (higher = more protection)")
        print(f"   Generated {len(self.generated_layer_params)} layer-specific parameters")
    
    def _create_layer_architecture(self) -> Dict[str, dict]:
        """Create architectural classification for pattern-based pruning."""
        
        # Get prunable layers
        prunable_layers = [name for name in self.model_state_dict.keys() 
                          if ('features' in name or 'classifier' in name) and 'weight' in name]
        
        architecture = {}
        
        for layer_name in prunable_layers:
            if layer_name in ['features.0.weight', 'features.4.weight']:
                arch_type = 'conv_early'
                base_protection = 85  # Very protective of early layers
                
            elif layer_name in ['features.8.weight', 'features.11.weight', 'features.15.weight']:
                arch_type = 'conv_mid' 
                base_protection = 65  # Moderate protection
                
            elif layer_name in ['features.18.weight', 'features.22.weight']:
                arch_type = 'conv_late'
                base_protection = 40  # Less protection, can prune more
                
            elif layer_name == 'classifier.0.weight':
                arch_type = 'fc_early'
                base_protection = 70  # First FC important for feature->hidden mapping
                
            elif layer_name == 'classifier.3.weight':
                arch_type = 'fc_mid'
                base_protection = 30  # Middle FC often has redundancy
                
            elif layer_name == 'classifier.6.weight':  
                arch_type = 'fc_final'
                base_protection = 50  # Final layer moderately important
                
            else:
                # Handle batch norm and other layers
                if 'features' in layer_name:
                    arch_type = 'conv_other'
                    base_protection = 60
                else:
                    arch_type = 'fc_other' 
                    base_protection = 40
            
            # Get sensitivity analysis if available
            guidance = self.sensitivity_analyzer.get_layer_guidance(layer_name)
            sensitivity_adjustment = 0
            
            if guidance and guidance.sensitivity_status == "VALID_SCORES":
                # Adjust protection based on actual sensitivity
                if guidance.actual_sensitivity_ratio > 0.8:
                    sensitivity_adjustment = 20  # More protection
                elif guidance.actual_sensitivity_ratio < 0.3:
                    sensitivity_adjustment = -15  # Less protection
            
            architecture[layer_name] = {
                'type': arch_type,
                'base_protection': base_protection,
                'sensitivity_adjustment': sensitivity_adjustment,
                'total_protection': base_protection + sensitivity_adjustment
            }
        
        return architecture
    
    def _generate_layer_parameters(self) -> List[float]:
        """Generate 19 layer-specific parameters from 4 pattern parameters."""
        
        layer_params = []
        
        # Get all prunable layers in consistent order
        prunable_layers = sorted([name for name in self.model_state_dict.keys() 
                                if ('features' in name or 'classifier' in name) and 'weight' in name])
        
        for layer_name in prunable_layers:
            arch_info = self.layer_architecture[layer_name]
            arch_type = arch_info['type']
            base_protection = arch_info['base_protection'] 
            total_protection = arch_info['total_protection']
            
            # Start with base sparsity
            target_sparsity = self.base_sparsity
            
            # Apply architectural bias
            if arch_type.startswith('conv'):
                target_sparsity += self.conv_bias
            else:  # FC layers
                target_sparsity += self.fc_bias
            
            # Apply protection based on architectural importance and sensitivity
            protection_factor = (total_protection / 100.0) * (self.protection_level / 100.0)
            protection_reduction = target_sparsity * protection_factor * 0.5  # Max 50% reduction
            
            final_sparsity = target_sparsity - protection_reduction
            
            # Apply architectural-specific bounds
            if arch_type == 'conv_early':
                final_sparsity = max(5.0, min(25.0, final_sparsity))   # Very conservative 
            elif arch_type == 'conv_mid':
                final_sparsity = max(10.0, min(50.0, final_sparsity))  # Moderate
            elif arch_type == 'conv_late': 
                final_sparsity = max(15.0, min(70.0, final_sparsity))  # More aggressive
            elif arch_type == 'fc_early':
                final_sparsity = max(10.0, min(60.0, final_sparsity))  # Moderate for first FC
            elif arch_type == 'fc_mid':
                final_sparsity = max(20.0, min(80.0, final_sparsity))  # Can be aggressive
            elif arch_type == 'fc_final':
                final_sparsity = max(15.0, min(75.0, final_sparsity))  # Moderate for output
            else:
                final_sparsity = max(10.0, min(70.0, final_sparsity))  # Default bounds
            
            layer_params.append(final_sparsity)
        
        return layer_params
    
    def generate_pruning_mask(self, device: torch.device = torch.device('cpu')) -> Dict[str, torch.Tensor]:
        """Generate pruning masks using the pattern-generated layer parameters."""
        
        from .improved_layerwise_agent import ImprovedLayerwisePruningAgent
        
        # Create improved agent with our generated parameters
        agent = ImprovedLayerwisePruningAgent(
            strategy_params=self.generated_layer_params,
            model_state_dict=self.model_state_dict,
            score_dir_path=self.score_dir_path,
            apply_constraints=True  # Let it apply safety constraints
        )
        
        print(f"🎭 Generating masks using architectural patterns...")
        print(f"   Pattern → Layer mapping:")
        
        prunable_layers = sorted([name for name in self.model_state_dict.keys() 
                                if ('features' in name or 'classifier' in name) and 'weight' in name])
        
        for i, layer_name in enumerate(prunable_layers):
            arch_info = self.layer_architecture[layer_name]
            param = self.generated_layer_params[i]
            print(f"      {layer_name} [{arch_info['type']}]: {param:.1f}%")
        
        return agent.generate_pruning_mask(device)
    
    def get_projected_sparsity(self, total_params: int) -> float:
        """Calculate projected overall sparsity."""
        
        # Estimate based on layer sizes and their pruning percentages
        total_pruned = 0
        total_weights = 0
        
        prunable_layers = sorted([name for name in self.model_state_dict.keys() 
                                if ('features' in name or 'classifier' in name) and 'weight' in name])
        
        for i, layer_name in enumerate(prunable_layers):
            layer_weights = self.model_state_dict[layer_name].numel()
            prune_percentage = self.generated_layer_params[i]
            
            total_weights += layer_weights
            total_pruned += layer_weights * (prune_percentage / 100.0)
        
        projected_sparsity = (total_pruned / total_weights) * 100.0 if total_weights > 0 else 0.0
        return projected_sparsity
    
    def get_pattern_summary(self) -> Dict:
        """Get summary of the pattern and generated parameters."""
        
        prunable_layers = sorted([name for name in self.model_state_dict.keys() 
                                if ('features' in name or 'classifier' in name) and 'weight' in name])
        
        # Group by architecture type
        type_params = {}
        for i, layer_name in enumerate(prunable_layers):
            arch_type = self.layer_architecture[layer_name]['type']
            param = self.generated_layer_params[i]
            
            if arch_type not in type_params:
                type_params[arch_type] = []
            type_params[arch_type].append(param)
        
        # Calculate statistics per type
        type_stats = {}
        for arch_type, params in type_params.items():
            type_stats[arch_type] = {
                'count': len(params),
                'mean': np.mean(params), 
                'min': np.min(params),
                'max': np.max(params),
                'std': np.std(params)
            }
        
        return {
            'pattern_params': {
                'base_sparsity': self.base_sparsity,
                'conv_bias': self.conv_bias, 
                'fc_bias': self.fc_bias,
                'protection_level': self.protection_level
            },
            'projected_sparsity': self.get_projected_sparsity(sum(p.numel() for p in self.model_state_dict.values())),
            'layer_count': len(self.generated_layer_params),
            'type_statistics': type_stats,
            'generated_params': self.generated_layer_params
        }
    
    def print_pattern_analysis(self):
        """Print detailed analysis of the architectural pattern."""
        
        summary = self.get_pattern_summary()
        
        print("\n" + "="*80)
        print("🏗️  ARCHITECTURAL PATTERN ANALYSIS") 
        print("="*80)
        
        print(f"📋 PATTERN PARAMETERS:")
        print(f"   Base Sparsity: {summary['pattern_params']['base_sparsity']:.1f}%")
        print(f"   Conv Bias: {summary['pattern_params']['conv_bias']:+.1f}%")
        print(f"   FC Bias: {summary['pattern_params']['fc_bias']:+.1f}%")
        print(f"   Protection Level: {summary['pattern_params']['protection_level']:.1f}%")
        
        print(f"\n📊 GENERATED LAYER STATISTICS:")
        print(f"   Total Layers: {summary['layer_count']}")
        print(f"   Projected Overall Sparsity: {summary['projected_sparsity']:.1f}%")
        
        print(f"\n🏛️  ARCHITECTURE TYPE BREAKDOWN:")
        for arch_type, stats in summary['type_statistics'].items():
            print(f"   {arch_type}: {stats['count']} layers, "
                  f"avg={stats['mean']:.1f}%, range={stats['min']:.1f}%-{stats['max']:.1f}%")
        
        print("="*80)


def create_pattern_individual_from_layerwise(layerwise_individual: List[float], 
                                           model_state_dict: Dict) -> List[float]:
    """Convert 19D layerwise individual to 4D pattern individual."""
    
    # Get layer architecture mapping
    prunable_layers = sorted([name for name in model_state_dict.keys() 
                            if ('features' in name or 'classifier' in name) and 'weight' in name])
    
    # Separate conv and FC parameters
    conv_params = []
    fc_params = []
    
    for i, layer_name in enumerate(prunable_layers):
        param = layerwise_individual[i]
        
        if 'features' in layer_name:
            conv_params.append(param)
        else:  # classifier
            fc_params.append(param)
    
    # Calculate pattern parameters
    base_sparsity = np.mean(layerwise_individual)  # Overall average
    
    if conv_params:
        conv_avg = np.mean(conv_params)
        conv_bias = conv_avg - base_sparsity
    else:
        conv_bias = 0.0
    
    if fc_params:
        fc_avg = np.mean(fc_params) 
        fc_bias = fc_avg - base_sparsity
    else:
        fc_bias = 0.0
    
    # Estimate protection level from variance (lower variance = higher protection)
    variance = np.var(layerwise_individual)
    protection_level = max(0.0, min(100.0, 100.0 - (variance / 10.0)))  # Heuristic mapping
    
    return [base_sparsity, conv_bias, fc_bias, protection_level]


def create_layerwise_individual_from_pattern(pattern_individual: List[float],
                                           model_state_dict: Dict,
                                           score_dir_path: str) -> List[float]:
    """Convert 4D pattern individual to 19D layerwise individual."""
    
    agent = ArchitecturalPatternAgent(pattern_individual, model_state_dict, score_dir_path)
    return agent.generated_layer_params