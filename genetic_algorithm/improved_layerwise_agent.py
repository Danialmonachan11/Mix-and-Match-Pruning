"""Improved layerwise pruning agent with robust sensitivity analysis and architectural awareness."""

import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .robust_sensitivity_analyzer import RobustSensitivityAnalyzer
from .agents import PruningStrategyAgent


class ImprovedLayerwisePruningAgent(PruningStrategyAgent):
    """
    Improved layerwise pruning agent that:
    1. Uses robust sensitivity analysis (handles overflow)
    2. Applies architectural constraints intelligently
    3. Provides clear feedback about why decisions were made
    4. Constrains GA parameters to safe ranges
    """
    
    def __init__(self, strategy_params: List[float], model_state_dict: Dict, 
                 score_dir_path: str, apply_constraints: bool = True):
        
        # Initialize parent class
        super().__init__(strategy_params, model_state_dict, score_dir_path)
        
        self.apply_constraints = apply_constraints
        
        # Initialize robust sensitivity analyzer
        print("🔧 Initializing Improved Layerwise Pruning Agent...")
        self.sensitivity_analyzer = RobustSensitivityAnalyzer(score_dir_path, model_state_dict)
        
        # Get constraint ranges for GA
        self.constraint_ranges = self.sensitivity_analyzer.get_constrained_pruning_ranges()
        
        # Apply constraints to strategy parameters if requested
        if apply_constraints:
            self.constrained_params = self._apply_intelligent_constraints()
        else:
            self.constrained_params = strategy_params.copy()
            
        print(f"🎯 Agent initialized: {len(self.prunable_layers)} layers, constraints={'ON' if apply_constraints else 'OFF'}")
    
    def _apply_intelligent_constraints(self) -> List[float]:
        """Apply intelligent constraints based on sensitivity analysis."""
        
        constrained_params = []
        constraint_log = []
        
        print("🛡️  Applying intelligent constraints based on sensitivity analysis...")
        
        for i, layer_name in enumerate(self.prunable_layers):
            original_param = self.strategy_params[i]
            
            # Get guidance for this layer
            guidance = self.sensitivity_analyzer.get_layer_guidance(layer_name)
            
            if guidance is None:
                # No guidance available, use original parameter
                constrained_params.append(original_param)
                constraint_log.append(f"{layer_name}: No guidance, using original {original_param:.1f}%")
                continue
            
            # Apply constraints
            min_safe = guidance.recommended_min_pruning
            max_safe = guidance.recommended_max_pruning
            
            if original_param < min_safe:
                constrained_param = min_safe
                constraint_log.append(f"{layer_name}: Increased {original_param:.1f}% → {constrained_param:.1f}% (below minimum)")
            elif original_param > max_safe:
                constrained_param = max_safe
                constraint_log.append(f"{layer_name}: Reduced {original_param:.1f}% → {constrained_param:.1f}% (above maximum)")
            else:
                constrained_param = original_param
                constraint_log.append(f"{layer_name}: Kept {original_param:.1f}% (within safe range)")
            
            constrained_params.append(constrained_param)
        
        # Print constraint summary
        changes = sum(1 for i, (orig, const) in enumerate(zip(self.strategy_params, constrained_params)) 
                     if abs(orig - const) > 0.1)
        
        if changes > 0:
            print(f"   ⚡ Applied constraints to {changes}/{len(self.prunable_layers)} layers:")
            for log_entry in constraint_log:
                if "→" in log_entry:  # Only show changes
                    print(f"      {log_entry}")
        else:
            print(f"   ✅ All {len(self.prunable_layers)} layers were within safe ranges")
        
        return constrained_params
    
    def generate_pruning_mask(self, device: torch.device = torch.device('cpu')) -> Dict[str, torch.Tensor]:
        """Generate pruning masks using constrained parameters and robust sensitivity analysis."""
        
        pruning_masks = {}
        
        print(f"🎭 Generating pruning masks with improved layerwise agent...")
        if self.apply_constraints:
            params_to_use = self.constrained_params
            print(f"   Using constrained parameters: {[f'{p:.1f}%' for p in params_to_use]}")
        else:
            params_to_use = self.strategy_params
            print(f"   Using original parameters: {[f'{p:.1f}%' for p in params_to_use]}")
        
        for i, layer_name in enumerate(self.prunable_layers):
            pruning_percentage = params_to_use[i]
            weights = self.model_state_dict[layer_name]
            guidance = self.sensitivity_analyzer.get_layer_guidance(layer_name)
            
            # Load sensitivity scores (using robust analyzer)
            score_file_path = self.layer_score_files.get(layer_name)
            
            if (guidance and guidance.sensitivity_status == "VALID_SCORES" and score_file_path):
                # Use actual sensitivity scores
                try:
                    scores_df = pd.read_csv(score_file_path)
                    raw_scores = scores_df['sensitivity_score'].astype(float).values
                    
                    # Use robust normalization that handles your overflow problem
                    sensitivity_scores = self._robust_score_normalization(raw_scores, layer_name)
                    sensitivity_scores = torch.from_numpy(sensitivity_scores).to(device)
                    
                    if weights.numel() != len(sensitivity_scores):
                        raise ValueError(f"Size mismatch: {weights.numel()} vs {len(sensitivity_scores)}")
                    
                    print(f"   📊 {layer_name}: Using VALID sensitivity scores (ratio: {guidance.actual_sensitivity_ratio:.2f})")
                    
                except Exception as e:
                    print(f"   ⚠️ {layer_name}: Error using sensitivity scores: {e}, falling back to magnitude")
                    sensitivity_scores = self._compute_magnitude_based_scores(weights, device)
            else:
                # Use magnitude-based scores as fallback
                sensitivity_scores = self._compute_magnitude_based_scores(weights, device)
                status = guidance.sensitivity_status if guidance else "NO_GUIDANCE"
                print(f"   📏 {layer_name}: Using magnitude-based scores ({status})")
            
            # Generate mask
            mask = self._create_sensitivity_based_mask(sensitivity_scores, pruning_percentage, device)
            pruning_masks[layer_name] = mask
            
            # Log results with guidance info
            num_pruned = (mask == 0).sum().item()
            total_weights = len(mask)
            actual_prune_percent = (num_pruned / total_weights) * 100 if total_weights > 0 else 0
            
            protection_level = guidance.protection_level if guidance else "UNKNOWN"
            arch_type = guidance.architectural_type if guidance else "unknown"
            
            print(f"   ✂️  {layer_name} [{protection_level}/{arch_type}]: "
                  f"{pruning_percentage:.1f}% → {num_pruned:,}/{total_weights:,} pruned ({actual_prune_percent:.1f}%)")
        
        return pruning_masks
    
    def _robust_score_normalization(self, raw_scores: np.ndarray, layer_name: str) -> np.ndarray:
        """Robust normalization that handles overflow and provides meaningful scores."""
        
        # Check for overflow (your main problem)
        max_abs_score = np.max(np.abs(raw_scores))
        
        if max_abs_score > 1e30:  # Clear overflow - use log-based normalization
            print(f"      🚨 Overflow detected in {layer_name} (max={max_abs_score:.2e}), using log normalization")
            
            # Log-based normalization for overflow cases
            finite_mask = np.isfinite(raw_scores)
            if not np.any(finite_mask):
                return np.ones_like(raw_scores) * 0.5  # All equal importance
            
            finite_scores = raw_scores[finite_mask]
            abs_finite_scores = np.abs(finite_scores)
            
            # Use log transformation to handle overflow
            log_scores = np.log10(abs_finite_scores + 1e-10)  # Add small constant to avoid log(0)
            
            # Normalize log scores to [0, 1]
            min_log = np.min(log_scores)
            max_log = np.max(log_scores)
            
            if max_log > min_log:
                normalized_finite = (log_scores - min_log) / (max_log - min_log)
            else:
                normalized_finite = np.ones_like(log_scores) * 0.5
            
            # Map back to full array
            result = np.ones_like(raw_scores) * np.median(normalized_finite)
            result[finite_mask] = normalized_finite
            
            return result
        
        else:
            # Standard normalization for reasonable values
            finite_mask = np.isfinite(raw_scores)
            if not np.any(finite_mask):
                return np.ones_like(raw_scores) * 0.5
            
            finite_scores = raw_scores[finite_mask]
            
            # Use robust percentile-based normalization
            p5, p95 = np.percentile(finite_scores, [5, 95])
            
            if p95 > p5:
                # Clip extreme values and normalize
                clipped_scores = np.clip(finite_scores, p5, p95)
                normalized_finite = (clipped_scores - p5) / (p95 - p5)
            else:
                normalized_finite = np.ones_like(finite_scores) * 0.5
            
            # Map back to full array  
            result = np.ones_like(raw_scores) * np.median(normalized_finite)
            result[finite_mask] = normalized_finite
            
            return result
    
    def _compute_magnitude_based_scores(self, weights: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Compute magnitude-based sensitivity scores as robust fallback."""
        
        # Simple magnitude-based scoring
        flattened_weights = weights.flatten().to(device)
        magnitude_scores = torch.abs(flattened_weights)
        
        # Normalize to [0, 1] range
        if magnitude_scores.max() > magnitude_scores.min():
            normalized_scores = (magnitude_scores - magnitude_scores.min()) / \
                              (magnitude_scores.max() - magnitude_scores.min())
        else:
            normalized_scores = torch.ones_like(magnitude_scores) * 0.5
        
        return normalized_scores
    
    def get_constraint_summary(self) -> Dict[str, any]:
        """Get summary of constraints applied."""
        
        if not self.apply_constraints:
            return {'constraints_applied': False}
        
        changes = []
        for i, layer_name in enumerate(self.prunable_layers):
            original = self.strategy_params[i]
            constrained = self.constrained_params[i]
            
            if abs(original - constrained) > 0.1:
                guidance = self.sensitivity_analyzer.get_layer_guidance(layer_name)
                changes.append({
                    'layer': layer_name,
                    'original': original,
                    'constrained': constrained,
                    'protection_level': guidance.protection_level if guidance else 'UNKNOWN',
                    'sensitivity_status': guidance.sensitivity_status if guidance else 'UNKNOWN'
                })
        
        return {
            'constraints_applied': True,
            'total_layers': len(self.prunable_layers),
            'layers_changed': len(changes),
            'changes': changes
        }
    
    def print_detailed_analysis(self):
        """Print detailed analysis of the agent's decision making."""
        print("\n" + "="*80)
        print("🔧 IMPROVED LAYERWISE AGENT ANALYSIS")
        print("="*80)
        
        # Print sensitivity analyzer results
        self.sensitivity_analyzer.print_comprehensive_analysis()
        
        # Print constraint summary
        constraint_summary = self.get_constraint_summary()
        
        if constraint_summary['constraints_applied']:
            print(f"\n🛡️  CONSTRAINT SUMMARY:")
            print(f"   Layers analyzed: {constraint_summary['total_layers']}")
            print(f"   Layers constrained: {constraint_summary['layers_changed']}")
            
            if constraint_summary['changes']:
                print(f"\n   📋 CONSTRAINT CHANGES:")
                for change in constraint_summary['changes']:
                    print(f"      {change['layer']}: {change['original']:.1f}% → {change['constrained']:.1f}% "
                          f"[{change['protection_level']}, {change['sensitivity_status']}]")
        else:
            print(f"\n🔓 CONSTRAINTS: Disabled - using raw GA parameters")
        
        print("="*80)