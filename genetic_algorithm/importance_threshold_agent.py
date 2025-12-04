"""
Phase 2 Implementation: Importance Threshold-Based Pruning Agent

This agent replaces percentage-based pruning with importance threshold-based pruning,
addressing the root cause of robustness gaps by respecting network structure.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from .agents import PruningStrategyAgent
import os
import glob
import pandas as pd


class ImportanceThresholdAgent(PruningStrategyAgent):
    """
    Phase 2 Agent: Uses importance thresholds instead of arbitrary percentages.
    
    Key Innovation: Evolves thresholds that work WITH sensitivity scores rather than 
    ignoring them. This preserves network structure like classical methods.
    """
    
    def __init__(self, model_state_dict: Dict, layer_score_files: Dict, 
                 prunable_layers: List[str], strategy_params: List[float],
                 architectural_constraints: Optional[Dict] = None):
        super().__init__(model_state_dict, layer_score_files, prunable_layers, strategy_params)
        self.architectural_constraints = architectural_constraints or {}
        
        # Convert strategy_params from percentage space [0-100] to threshold space [0-1]
        # This maintains compatibility with existing GA infrastructure
        self.importance_thresholds = [self._convert_param_to_threshold(param) for param in strategy_params]
        
        print(f"ImportanceThresholdAgent initialized with {len(self.importance_thresholds)} thresholds")
        print(f"  Thresholds: {[f'{t:.3f}' for t in self.importance_thresholds]}")
    
    def _convert_param_to_threshold(self, param: float) -> float:
        """
        Convert GA parameter (0-100 range) to importance threshold (0-1 range).
        
        Higher GA parameter -> Higher threshold -> More pruning (only keep very important weights)
        Lower GA parameter -> Lower threshold -> Less pruning (keep more weights)
        """
        # Clamp parameter to valid range
        param = max(0.1, min(param, 99.9))
        
        # Convert to threshold: higher param = higher threshold = more aggressive
        threshold = param / 100.0
        
        return threshold
    
    def generate_pruning_mask(self, device: torch.device = None) -> Dict[str, torch.Tensor]:
        """
        Generate pruning masks based on importance thresholds rather than percentages.
        
        This is the core Phase 2 innovation: prune based on importance rather than quantity.
        """
        if device is None:
            device = torch.device('cpu')
        
        pruning_masks = {}
        total_weights = 0
        total_pruned = 0
        
        print(f"Generating importance-based masks with thresholds: {[f'{t:.3f}' for t in self.importance_thresholds]}")
        
        for i, layer_name in enumerate(self.prunable_layers):
            importance_threshold = self.importance_thresholds[i]
            
            weights = self.model_state_dict[layer_name]
            
            # Calculate importance scores (combines sensitivity + magnitude + position)
            importance_scores = self._calculate_comprehensive_importance_scores(
                weights, layer_name
            )
            
            # Apply architectural constraints
            constrained_threshold = self._apply_architectural_constraints(
                importance_threshold, layer_name, i
            )
            
            # Generate mask based on importance threshold
            mask = self._create_importance_threshold_mask(
                importance_scores, constrained_threshold, device
            )
            
            pruning_masks[layer_name] = mask
            
            # Statistics
            num_pruned = (mask == 0).sum().item()
            layer_total = len(mask)
            actual_prune_percent = (num_pruned / layer_total) * 100 if layer_total > 0 else 0
            
            total_weights += layer_total
            total_pruned += num_pruned
            
            print(f"  {layer_name}: threshold={constrained_threshold:.3f} -> "
                  f"{num_pruned}/{layer_total} pruned ({actual_prune_percent:.1f}%)")
        
        overall_sparsity = (total_pruned / total_weights) * 100 if total_weights > 0 else 0
        print(f"  Overall sparsity achieved: {overall_sparsity:.1f}%")
        
        return pruning_masks
    
    def _calculate_comprehensive_importance_scores(self, weights: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Calculate comprehensive importance scores combining multiple factors.
        
        This replaces the arbitrary percentage approach with structured importance analysis.
        """
        weights_flat = weights.view(-1)
        
        # Component 1: Magnitude-based importance (like classical methods)
        magnitude_scores = torch.abs(weights_flat)
        
        # Component 2: Sensitivity-based importance (from CSV files)
        sensitivity_scores = self._get_layer_sensitivity_scores(weights_flat, layer_name)
        
        # Component 3: Position-based importance (architectural knowledge)
        position_scores = self._get_position_importance(weights, layer_name)
        
        # Component 4: Flow preservation importance (new Phase 2 feature)
        flow_scores = self._get_flow_preservation_scores(weights, layer_name)
        
        # Combine and normalize all components
        combined_scores = self._combine_importance_components(
            magnitude_scores, sensitivity_scores, position_scores, flow_scores, layer_name
        )
        
        return combined_scores
    
    def _get_layer_sensitivity_scores(self, weights_flat: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Load and process sensitivity scores from CSV files."""
        score_file_path = self.layer_score_files.get(layer_name)
        
        if score_file_path and os.path.exists(score_file_path):
            try:
                df = pd.read_csv(score_file_path)
                if 'sensitivity_score' in df.columns:
                    raw_scores = df['sensitivity_score'].astype(float).values
                    
                    # Convert to tensor and match weights shape
                    sensitivity_tensor = torch.tensor(raw_scores, dtype=torch.float32)
                    
                    if len(sensitivity_tensor) == len(weights_flat):
                        # Normalize to [0, 1] range
                        min_val, max_val = sensitivity_tensor.min(), sensitivity_tensor.max()
                        if max_val > min_val:
                            normalized = (sensitivity_tensor - min_val) / (max_val - min_val)
                        else:
                            normalized = torch.ones_like(sensitivity_tensor) * 0.5
                        
                        return normalized.to(weights_flat.device)
            except Exception as e:
                print(f"    Warning: Could not load sensitivity for {layer_name}: {e}")
        
        # Fallback: Use magnitude as proxy for sensitivity
        magnitude = torch.abs(weights_flat)
        min_val, max_val = magnitude.min(), magnitude.max()
        if max_val > min_val:
            return (magnitude - min_val) / (max_val - min_val)
        else:
            return torch.ones_like(weights_flat) * 0.5
    
    def _get_position_importance(self, weights: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Calculate position-based importance scores for network structure preservation."""
        weights_flat = weights.view(-1)
        
        if 'features' in layer_name and len(weights.shape) == 4:
            # For conv layers: output_channels x input_channels x height x width
            # Create position-based importance (center weights more important)
            out_ch, in_ch, h, w = weights.shape
            
            # Create spatial importance map (center of kernel more important)
            h_center, w_center = h // 2, w // 2
            spatial_importance = torch.ones((h, w))
            for i in range(h):
                for j in range(w):
                    dist_from_center = abs(i - h_center) + abs(j - w_center)
                    spatial_importance[i, j] = 1.0 / (1.0 + dist_from_center * 0.1)
            
            # Broadcast to full weight tensor shape and flatten
            full_importance = spatial_importance.unsqueeze(0).unsqueeze(0).expand(out_ch, in_ch, -1, -1)
            return full_importance.reshape(-1).to(weights.device)
        else:
            # For FC layers: uniform importance
            return torch.ones_like(weights_flat)
    
    def _get_flow_preservation_scores(self, weights: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        NEW Phase 2 Feature: Calculate flow preservation importance scores.
        
        This ensures pruning preserves information flow through the network.
        """
        weights_flat = weights.view(-1)
        
        # Extract layer index for flow analysis
        layer_idx = self._extract_layer_index(layer_name)
        
        if 'features' in layer_name:
            # For conv layers: preserve diverse filter patterns
            if len(weights.shape) == 4:  # Conv weights: [out_ch, in_ch, h, w]
                out_ch, in_ch, h, w = weights.shape
                
                # Calculate filter diversity (how unique each filter is)
                filters = weights.reshape(out_ch, -1)  # Flatten each filter
                
                # Measure diversity of each weight's contribution across filters
                weight_importance = torch.zeros_like(weights_flat)
                
                for i in range(out_ch):
                    filter_weights = filters[i].abs()
                    # Weights that contribute to diverse filters are more important
                    diversity_score = filter_weights / (filter_weights.sum() + 1e-8)
                    
                    # Map back to full weight tensor
                    start_idx = i * (in_ch * h * w)
                    end_idx = start_idx + (in_ch * h * w)
                    weight_importance[start_idx:end_idx] += diversity_score
                
                return weight_importance / out_ch  # Normalize by number of filters
        
        elif 'classifier' in layer_name:
            # For FC layers: preserve input diversity
            if len(weights.shape) == 2:  # FC weights: [out_features, in_features]
                out_features, in_features = weights.shape
                
                # Input connections that preserve information diversity
                input_diversity = torch.zeros(in_features)
                for j in range(in_features):
                    # How diverse are the outputs this input contributes to
                    output_contributions = weights[:, j].abs()
                    diversity = torch.std(output_contributions) + torch.mean(output_contributions)
                    input_diversity[j] = diversity
                
                # Broadcast to full weight shape
                flow_importance = input_diversity.unsqueeze(0).expand(out_features, -1).reshape(-1)
                
                # Normalize
                min_val, max_val = flow_importance.min(), flow_importance.max()
                if max_val > min_val:
                    return (flow_importance - min_val) / (max_val - min_val)
        
        # Fallback: uniform importance
        return torch.ones_like(weights_flat) * 0.5
    
    def _combine_importance_components(self, magnitude_scores: torch.Tensor, 
                                     sensitivity_scores: torch.Tensor,
                                     position_scores: torch.Tensor,
                                     flow_scores: torch.Tensor,
                                     layer_name: str) -> torch.Tensor:
        """
        Combine different importance components with layer-specific weights.
        
        This creates a comprehensive importance metric that classical methods
        naturally approximate but GA can optimize.
        """
        
        # Layer-specific component weights
        if 'features.0' in layer_name or 'features.1' in layer_name:
            # Early layers: heavily weight sensitivity (structure preservation)
            weights_config = {'magnitude': 0.2, 'sensitivity': 0.5, 'position': 0.2, 'flow': 0.1}
        elif 'features' in layer_name:
            # Middle conv layers: balance all components
            weights_config = {'magnitude': 0.3, 'sensitivity': 0.3, 'position': 0.2, 'flow': 0.2}
        elif 'classifier.0' in layer_name:
            # First FC layer: emphasize flow preservation
            weights_config = {'magnitude': 0.3, 'sensitivity': 0.2, 'position': 0.1, 'flow': 0.4}
        else:
            # Later layers: more magnitude-focused
            weights_config = {'magnitude': 0.4, 'sensitivity': 0.2, 'position': 0.1, 'flow': 0.3}
        
        # Normalize all components to [0, 1] range
        def normalize_component(scores):
            min_val, max_val = scores.min(), scores.max()
            if max_val > min_val:
                return (scores - min_val) / (max_val - min_val)
            else:
                return torch.ones_like(scores) * 0.5
        
        magnitude_norm = normalize_component(magnitude_scores)
        sensitivity_norm = normalize_component(sensitivity_scores)
        position_norm = normalize_component(position_scores)
        flow_norm = normalize_component(flow_scores)
        
        # Weighted combination
        combined_importance = (
            weights_config['magnitude'] * magnitude_norm +
            weights_config['sensitivity'] * sensitivity_norm +
            weights_config['position'] * position_norm +
            weights_config['flow'] * flow_norm
        )
        
        return combined_importance
    
    def _apply_architectural_constraints(self, threshold: float, layer_name: str, layer_idx: int) -> float:
        """
        Apply architectural constraints to prevent structural damage.
        
        This is Phase 2's answer to the early layer protection problem.
        """
        constrained_threshold = threshold
        
        # Early layer protection: lower thresholds = less pruning
        if 'features.0' in layer_name:
            # Most critical layer: maximum threshold of 0.1 (keep 90%+ of weights)
            constrained_threshold = min(threshold, 0.1)
        elif 'features.1' in layer_name or 'features.2' in layer_name:
            # Critical layers: maximum threshold of 0.2
            constrained_threshold = min(threshold, 0.2)
        elif layer_idx < 4:  # First 4 layers
            # Important early layers: maximum threshold of 0.3
            constrained_threshold = min(threshold, 0.3)
        
        # Classifier layer constraints
        elif 'classifier.0' in layer_name:
            # First FC layer: moderate constraint
            constrained_threshold = min(threshold, 0.4)
        
        # Custom constraints from configuration
        if layer_name in self.architectural_constraints:
            max_threshold = self.architectural_constraints[layer_name]
            constrained_threshold = min(constrained_threshold, max_threshold)
        
        return constrained_threshold
    
    def _create_importance_threshold_mask(self, importance_scores: torch.Tensor, 
                                        threshold: float, device: torch.device) -> torch.Tensor:
        """
        Create pruning mask based on importance threshold.
        
        This is the core Phase 2 logic: keep weights above threshold.
        """
        # Keep weights with importance above threshold
        mask = (importance_scores > threshold).float().to(device)
        
        # Ensure we don't prune everything (minimum 10% kept)
        num_kept = mask.sum().item()
        total_weights = len(mask)
        keep_ratio = num_kept / total_weights
        
        if keep_ratio < 0.1:  # Less than 10% kept
            # Keep top 10% by importance
            num_to_keep = max(1, int(total_weights * 0.1))
            _, top_indices = torch.topk(importance_scores, num_to_keep)
            mask = torch.zeros_like(importance_scores)
            mask[top_indices] = 1.0
        
        return mask.to(device)
    
    def _extract_layer_index(self, layer_name: str) -> int:
        """Extract layer index from layer name for architectural analysis."""
        try:
            # Extract number from names like "features.0.weight" -> 0
            parts = layer_name.split('.')
            for part in parts:
                if part.isdigit():
                    return int(part)
        except:
            pass
        return 0
    
    def get_projected_sparsity(self, original_total_params: int) -> float:
        """
        Estimate projected sparsity based on thresholds.
        
        Note: This is approximate since actual pruning depends on importance distributions.
        """
        # Use threshold as rough proxy for sparsity (higher threshold = more pruning)
        avg_threshold = sum(self.importance_thresholds) / len(self.importance_thresholds)
        estimated_sparsity = avg_threshold * 70.0  # Rough estimate
        
        return min(estimated_sparsity, 90.0)  # Cap at 90%