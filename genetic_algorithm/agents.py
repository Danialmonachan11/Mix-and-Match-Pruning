"""Pruning strategy agents for genetic algorithm."""

import copy
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from torch.amp import autocast, GradScaler # Use new torch.amp API
from core.utils import (
    get_layer_score_files_map, count_nonzero_parameters, 
    measure_latency, test_accuracy, cleanup_memory
)
from benchmarking.reliability.fault_injection import FaultInjector


def get_sensitivity_based_layer_ordering(available_layers: List[str], score_dir_path: str) -> List[str]:
    """Get layers ordered by sensitivity (least to most sensitive) to match initialization ordering."""
    layer_rankings = []

    for layer_name in available_layers:
        try:
            # Try to load sensitivity scores to rank layers
            import glob
            pattern = os.path.join(score_dir_path, f"weight_sensitivity_scores_{layer_name}.csv")
            files = glob.glob(pattern)

            if files:
                df = pd.read_csv(files[0])
                if 'sensitivity_score' in df.columns:
                    scores = df['sensitivity_score'].astype(float).values
                    avg_sensitivity = np.mean(np.abs(scores))
                    layer_rankings.append((avg_sensitivity, layer_name))
                else:
                    # Fallback: assume high sensitivity
                    layer_rankings.append((1.0, layer_name))
            else:
                # Fallback: assume high sensitivity
                layer_rankings.append((1.0, layer_name))

        except Exception as e:
            print(f"Warning: Error loading sensitivity for {layer_name}: {e}")
            layer_rankings.append((1.0, layer_name))

    # Sort by average sensitivity (least sensitive first) - SAME AS INITIALIZATION
    layer_rankings.sort(key=lambda x: x[0])

    print("Agent layer ordering (least → most sensitive):")
    for rank, (avg_sens, layer_name) in enumerate(layer_rankings):
        print(f"  {rank+1}. {layer_name}: avg_sensitivity={avg_sens:.6e}")

    return [layer_name for _, layer_name in layer_rankings]


class PruningStrategyAgent:
    """Agent that generates pruning masks based on strategy parameters."""
    
    def __init__(self, strategy_params: List[float], model_state_dict: Dict,
                 score_dir_path: str):

        self.strategy_params = strategy_params
        self.model_state_dict = model_state_dict
        self.score_dir_path = score_dir_path
        self.layer_score_files = get_layer_score_files_map(score_dir_path, model_state_dict)
        
        # Get prunable layers that have sensitivity scores - USE SENSITIVITY-BASED ORDERING
        available_layers = [
            name for name in model_state_dict.keys()
            if ('features' in name or 'classifier' in name) and 'weight' in name
            and name in self.layer_score_files
        ]

        # Use sensitivity-based ordering to match initialization
        self.prunable_layers = get_sensitivity_based_layer_ordering(available_layers, score_dir_path)
        
        if len(strategy_params) != len(self.prunable_layers):
            raise ValueError(
                f"Strategy params length ({len(strategy_params)}) must match "
                f"prunable layers length ({len(self.prunable_layers)})"
            )
    
    def generate_pruning_mask(self, device: torch.device = torch.device('cpu')) -> Dict[str, torch.Tensor]:
       
        pruning_masks = {}
        
        print(f"Generating masks with percentiles: {[f'{p:.1f}%' for p in self.strategy_params]}")
        
        for i, layer_name in enumerate(self.prunable_layers):
            # Clamp percentile to safe range
            # Allow 0.0% for complete layer protection in sequential approach
            effective_percentile = max(0.0, min(self.strategy_params[i], 99.9))
            
            weights = self.model_state_dict[layer_name]
            score_file_path = self.layer_score_files.get(layer_name)
            
            if not score_file_path:
                print(f"Warning: No score file for {layer_name}. Using improved sensitivity.")
                sensitivity_scores = self._compute_improved_sensitivity(layer_name, weights)
            else:
                try:
                    # Load sensitivity scores
                    scores_df = pd.read_csv(score_file_path)
                    raw_scores = scores_df['sensitivity_score'].astype(float).values
                    normalized_scores = self._normalize_sensitivity_scores(raw_scores)
                    print(f"DEBUG {layer_name}: raw_max={np.max(raw_scores):.2e}, normalized_max={np.max(normalized_scores):.6f}")
                    sensitivity_scores = torch.from_numpy(normalized_scores).to(device)

                    # Validate dimensions first
                    if weights.numel() != len(sensitivity_scores):
                        print(f"Warning: Dimension mismatch for {layer_name}. Using improved sensitivity.")
                        sensitivity_scores = self._compute_improved_sensitivity(layer_name, weights)
                    # Check if sensitivity scores are broken and use improved scoring if needed
                    elif self._scores_are_broken(sensitivity_scores, layer_name):
                        print(f"Computing improved sensitivity scores for {layer_name}")
                        sensitivity_scores = self._compute_improved_sensitivity(layer_name, weights)
                    else:
                        print(f"Using pre-computed sensitivity scores for {layer_name}")

                except Exception as e:
                    print(f"Error loading scores for {layer_name}: {e}")
                    print(f"Falling back to improved sensitivity computation")
                    sensitivity_scores = self._compute_improved_sensitivity(layer_name, weights)

            # Get weight magnitudes for vulnerability-based pruning
            weight_magnitudes = torch.abs(weights.view(-1)).to(device)

            # Generate mask using MENTOR'S VULNERABILITY-BASED APPROACH
            mask = self._create_sensitivity_based_mask(
                sensitivity_scores, effective_percentile, device, weight_magnitudes
            )
            pruning_masks[layer_name] = mask
            
            # Log pruning statistics
            num_pruned = (mask == 0).sum().item()
            total_weights = len(mask)
            actual_prune_percent = (num_pruned / total_weights) * 100 if total_weights > 0 else 0
            
            print(f"  {layer_name}: {effective_percentile:.1f}% -> "
                  f"{num_pruned}/{total_weights} pruned ({actual_prune_percent:.1f}%)")
        
        return pruning_masks
    
    def _create_sensitivity_based_mask(self, sensitivity_scores: torch.Tensor,
                                     percentile: float, device: torch.device,
                                     weight_magnitudes: torch.Tensor = None) -> torch.Tensor:
        """
        Create pruning mask using RELIABILITY-AWARE IMPORTANCE SCORING.

        APPROACH: Product-based importance = Sensitivity × Magnitude
        - High S × High M = HIGH importance → KEEP (accurate AND fault-robust)
        - High S × Low M = MEDIUM importance → KEEP if possible (accurate but fragile)
        - Low S × High M = MEDIUM importance → KEEP if possible (robust but less critical)
        - Low S × Low M = LOW importance → PRUNE (neither accurate nor robust)

        This approach:
        1. Preserves accuracy by protecting high-sensitivity weights
        2. Improves reliability by preferring high-magnitude weights
        3. Avoids division issues and normalization artifacts
        """
        total_weights = len(sensitivity_scores)
        num_to_prune = int(total_weights * (percentile / 100.0))

        if num_to_prune <= 0:
            return torch.ones(total_weights, device=device).float()

        # RELIABILITY-AWARE IMPORTANCE SCORING
        if weight_magnitudes is not None and len(weight_magnitudes) == len(sensitivity_scores):
            # Normalize both to [0,1] for fair comparison
            sens_min, sens_max = sensitivity_scores.min(), sensitivity_scores.max()
            mag_min, mag_max = weight_magnitudes.min(), weight_magnitudes.max()

            sens_norm = (sensitivity_scores - sens_min) / (sens_max - sens_min + 1e-10)
            mag_norm = (weight_magnitudes - mag_min) / (mag_max - mag_min + 1e-10)

            # NOVEL MULTI-CRITERIA SCORING
            # Combines accuracy + robustness + vulnerability with tunable weights
            α = 0.5  # Accuracy importance weight
            β = 0.3  # Fault robustness weight
            γ = 0.2  # Vulnerability penalty weight

            # 1. Accuracy importance (keep high-sensitivity weights)
            accuracy_score = sens_norm

            # 2. Fault robustness (prefer high-magnitude weights - resistant to bit flips)
            robustness_score = mag_norm

            # 3. Fault vulnerability (penalize high S/low M - fragile under faults)
            vulnerability_score = sens_norm / (mag_norm + 0.1)

            # FINAL SCORE: Higher = keep, Lower = prune
            pruning_score = (α * accuracy_score) + (β * robustness_score) - (γ * vulnerability_score)

            # Prune LOWEST scores (safe to remove)
            indices_to_prune = torch.argsort(pruning_score, descending=False)[:num_to_prune]

            print(f"    MULTI-CRITERIA MODE: Pruning {num_to_prune} weights (α={α} β={β} γ={γ})")
        else:
            # Fallback: sensitivity-only pruning (prune lowest sensitivity)
            print(f"    WARNING: No magnitudes provided, using sensitivity-only pruning")
            indices_to_prune = torch.argsort(sensitivity_scores, descending=False)[:num_to_prune]

        # Create mask
        mask = torch.ones(total_weights, device=device).float()
        mask[indices_to_prune] = 0.0

        return mask
    
    def _normalize_sensitivity_scores(self, scores: np.ndarray) -> np.ndarray:
        """IMPROVED: Log-scale normalization preserves magnitude relationships for extreme values."""
        finite_mask = np.isfinite(scores)
        if not np.any(finite_mask):
            return np.ones_like(scores) * 0.5
        
        finite_scores = scores[finite_mask]
        abs_scores = np.abs(finite_scores)
        
        # Handle edge cases
        if len(np.unique(abs_scores)) <= 1:
            return np.ones_like(scores) * 0.5
        
        # IMPROVED: Use log-scale normalization for extreme values (like 1e37)
        if np.max(abs_scores) > 1e10:  # High dynamic range detected
            # Add small epsilon to handle zeros
            min_nonzero = np.min(abs_scores[abs_scores > 0]) if np.any(abs_scores > 0) else 1e-15
            epsilon = min_nonzero * 1e-3
            log_scores = np.log10(abs_scores + epsilon)
            
            # Min-max normalization in log space preserves relative differences
            log_min, log_max = np.min(log_scores), np.max(log_scores)
            if log_max > log_min:
                normalized = (log_scores - log_min) / (log_max - log_min)
            else:
                normalized = np.ones_like(log_scores) * 0.5
        else:
            # Standard min-max for reasonable ranges
            min_val, max_val = np.min(abs_scores), np.max(abs_scores)
            if max_val > min_val:
                normalized = (abs_scores - min_val) / (max_val - min_val)
            else:
                normalized = np.ones_like(abs_scores) * 0.5
        
        # Map back to full array
        result = np.ones_like(scores) * 0.5
        result[finite_mask] = normalized
        
        return result

    def get_projected_sparsity(self, original_total_params: int) -> float:
       
        total_pruned = 0
        
        for i, layer_name in enumerate(self.prunable_layers):
            layer_params = self.model_state_dict[layer_name].numel()
            # Allow 0.0% for complete layer protection in sequential approach
            effective_percentile = max(0.0, min(self.strategy_params[i], 99.9))
            layer_pruned = int(layer_params * (effective_percentile / 100.0))
            total_pruned += layer_pruned
        
        return (total_pruned / original_total_params) * 100.0 if original_total_params > 0 else 0.0
    
    def _scores_are_broken(self, sensitivity_scores: torch.Tensor, layer_name: str) -> bool:
        """Detect if sensitivity scores are broken (overflow/underflow/all zeros)."""
        if len(sensitivity_scores) == 0:
            return True
            
        # IMPROVED: Only check for mathematical invalidity, not extreme values
        has_inf = torch.isinf(sensitivity_scores).any()
        has_nan = torch.isnan(sensitivity_scores).any()
        
        # REMOVED: has_overflow check - high values (1e37) indicate important weights!
        # REMOVED: all_zeros check - zero layers should get uniform low importance
        
        # IMPROVED: Only reject if ALL scores are identical (no ranking possible)
        unique_vals = torch.unique(sensitivity_scores)
        completely_uniform = len(unique_vals) == 1  # More permissive than <= 1
        
        is_broken = has_inf or has_nan or completely_uniform
        
        if is_broken:
            print(f"    Broken sensitivity detected: inf={has_inf}, nan={has_nan}, "
                  f"uniform={completely_uniform} (unique_vals={len(unique_vals)})")
        else:
            print(f"    Sensitivity scores accepted: {len(unique_vals)} unique values, "
                  f"max={sensitivity_scores.max().item():.2e}")
        
        return is_broken
    
    def _extract_layer_index(self, layer_name: str) -> int:
        """Extract layer index for importance weighting."""
        try:
            # Extract number from layer name like "features.0.weight" -> 0
            parts = layer_name.split('.')
            for part in parts:
                if part.isdigit():
                    return int(part)
        except:
            pass
        return 999  # Default high index if parsing fails
    
    def _compute_improved_sensitivity(self, layer_name: str, weights: torch.Tensor) -> torch.Tensor:
        """Compute improved sensitivity scores when original scores are broken."""
        device = weights.device
        weights_flat = weights.view(-1)
        
        # Method 1: Magnitude-based scoring (reliable baseline)
        magnitude_scores = torch.abs(weights_flat)
        
        # Method 2: Layer-type adaptive scoring
        layer_scores = self._get_layer_adaptive_scores(layer_name, weights_flat)
        
        # Method 3: Position-based importance (for conv layers)
        position_scores = self._get_position_importance(weights, layer_name)
        
        # Combine scores with proper normalization
        combined_scores = self._combine_and_normalize_scores(
            magnitude_scores, layer_scores, position_scores, layer_name
        )
        
        return combined_scores
    
    def _get_layer_adaptive_scores(self, layer_name: str, weights_flat: torch.Tensor) -> torch.Tensor:
        """Different scoring strategies based on layer type."""
        base_magnitude = torch.abs(weights_flat)
        
        if 'features' in layer_name:
            # Convolutional layers: Use magnitude with early-layer protection
            layer_idx = self._extract_layer_index(layer_name)
            
            # Early conv layers (0-2) are more critical
            if layer_idx <= 2:
                importance_multiplier = 2.0  # 2x importance for early layers
            elif layer_idx <= 6:
                importance_multiplier = 1.5  # 1.5x importance for middle layers  
            else:
                importance_multiplier = 1.0  # Normal importance for late layers
                
            return base_magnitude * importance_multiplier
            
        elif 'classifier' in layer_name:
            # Fully connected layers: Prevent all-zero issue with minimum floor
            min_score = base_magnitude.max() * 0.05  # 5% of max as minimum
            protected_scores = torch.maximum(base_magnitude, 
                                           torch.full_like(base_magnitude, min_score))
            
            # First classifier layer is most critical
            layer_idx = self._extract_layer_index(layer_name) 
            if layer_idx == 0:  # classifier.0 (features -> first FC)
                return protected_scores * 1.8  # Extra protection
            else:
                return protected_scores * 1.2  # Standard protection
                
        else:
            # Default case: pure magnitude
            return base_magnitude
    
    def _get_position_importance(self, weights: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Assign importance based on weight position (useful for conv layers)."""
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
    
    def _combine_and_normalize_scores(self, magnitude_scores: torch.Tensor, 
                                    layer_scores: torch.Tensor, 
                                    position_scores: torch.Tensor,
                                    layer_name: str) -> torch.Tensor:
        """Combine different scoring methods with proper normalization."""
        
        # Normalize each component to [0, 1] range using min-max scaling
        def normalize_scores(scores):
            min_val = scores.min()
            max_val = scores.max()
            if max_val > min_val:
                return (scores - min_val) / (max_val - min_val)
            else:
                return torch.ones_like(scores) * 0.5  # Fallback if all identical
        
        mag_norm = normalize_scores(magnitude_scores)
        layer_norm = normalize_scores(layer_scores)  
        pos_norm = normalize_scores(position_scores)
        
        # Layer-type specific combination weights
        if 'classifier' in layer_name:
            # Classifier layers: Emphasize magnitude and layer-specific scoring
            combined = 0.6 * mag_norm + 0.35 * layer_norm + 0.05 * pos_norm
        elif 'features' in layer_name:
            # Convolutional layers: Balanced combination with position importance
            combined = 0.4 * mag_norm + 0.4 * layer_norm + 0.2 * pos_norm
        else:
            # Default: Magnitude-focused
            combined = 0.7 * mag_norm + 0.2 * layer_norm + 0.1 * pos_norm
        
        # Final normalization and conversion to meaningful range
        final_scores = normalize_scores(combined)
        
        # Add small epsilon to prevent exact zeros (helps with sorting stability)
        epsilon = 1e-8
        final_scores = final_scores + epsilon
        
        return final_scores


class ModelPruningAgent:
    """Agent that applies pruning masks to create pruned models."""
    
    def prune_model(self, model_instance: nn.Module, 
                   pruning_masks: Dict[str, torch.Tensor]) -> nn.Module:
       
        pruned_model = copy.deepcopy(model_instance)
        
        with torch.no_grad():
            for layer_name, mask in pruning_masks.items():
                if layer_name in pruned_model.state_dict():
                    weights = pruned_model.state_dict()[layer_name]
                    if weights.numel() == mask.numel():
                        # Apply mask
                        weights.data.view(-1).mul_(mask.to(weights.device))
                    else:
                        print(f"Warning: Shape mismatch for {layer_name}. "
                              f"Model: {weights.numel()}, Mask: {mask.numel()}. Skipping.")
        
        return pruned_model
    
    def enforce_sparsity(self, model: nn.Module, pruning_masks: Dict[str, torch.Tensor]) -> None:
       
        with torch.no_grad():
            for layer_name, mask in pruning_masks.items():
                if layer_name in model.state_dict():
                    weights = model.state_dict()[layer_name]
                    if weights.numel() == mask.numel():
                        weights.data.view(-1).mul_(mask.to(weights.device))


class EvaluationAgent:
    """Agent for evaluating model performance across multiple metrics."""
    
    def __init__(self, dataloader, device: torch.device):
     
        self.dataloader = dataloader
        self.device = device
    
    def evaluate(self, model_instance: nn.Module, dummy_input: Optional[torch.Tensor] = None) -> Dict:
        
        # Test accuracy
        accuracy = test_accuracy(model_instance, self.device, self.dataloader)
        
        # Calculate sparsity (parameter-weighted - correct method)
        num_nonzero_params = count_nonzero_parameters(model_instance)
        total_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
        sparsity = ((total_params - num_nonzero_params) / total_params * 100) if total_params > 0 else 0
        
        # Measure latency
        latency = 0.0
        if dummy_input is not None:
            try:
                latency = measure_latency(model_instance, dummy_input)
            except Exception as e:
                print(f"Warning: Latency measurement failed: {e}")
        
        results = {
            "accuracy": accuracy,
            "sparsity": sparsity,
            "latency": latency,
            "nonzero_params": num_nonzero_params,
            "total_params": total_params
        }
        
        print(f"Evaluation: Acc={accuracy:.2f}%, Sparsity={sparsity:.2f}%, "
              f"Latency={latency:.3f}ms, Params={num_nonzero_params}/{total_params}")
        
        return results
    
    def quick_evaluate(self, model_instance: nn.Module) -> float:
        
        return test_accuracy(model_instance, self.device, self.dataloader)


class ReliabilityAwareEvaluationAgent(EvaluationAgent):
    """Extended evaluation agent with reliability assessment."""
    
    def __init__(self, dataloader, device: torch.device, reliability_tester=None):
        
        super().__init__(dataloader, device)
        self.reliability_tester = reliability_tester
    
    def evaluate_with_reliability(self, model_instance: nn.Module, 
                                dummy_input: Optional[torch.Tensor] = None,
                                num_faults: int = 100, repetitions: int = 10) -> Dict:
       
        # Get basic evaluation metrics
        results = self.evaluate(model_instance, dummy_input)
        
        # Add reliability assessment
        if self.reliability_tester:
            try:
                reliability_score = self.reliability_tester.quick_reliability_estimate(
                    model_instance, num_faults, repetitions
                )
                results["reliability"] = reliability_score
                print(f"  Reliability: {reliability_score:.2f}% (avg acc with {num_faults} faults)")
            except Exception as e:
                print(f"Warning: Reliability assessment failed: {e}")
                results["reliability"] = 0.0
        else:
            results["reliability"] = 0.0
        
        return results


class FineTuningAgent:
    """Agent for fine-tuning pruned models."""
    
    def __init__(self, train_dataloader, val_dataloader, device: torch.device):
       
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        # Fine-tuning focuses on accuracy recovery only

    def finetune(self, model: nn.Module, pruning_masks: Dict[str, torch.Tensor],
                epochs: int = 20, lr: float = 1e-4, patience: int = 3, enable_finetuning: bool = False) -> nn.Module:
       
        if not enable_finetuning:
            print(f"Fine-tuning disabled to evaluate inherent quality of pruning strategies")
            
            # Simply return the model without fine-tuning
            # This ensures we evaluate the inherent quality of the pruning strategy
            # without any compensation from fine-tuning
            target_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            target_model.eval()
            
            return target_model
        
        # FINE-TUNING IMPLEMENTATION (ENABLED FOR LAYER-WISE MODE)
        print(f"Reliability-aware fine-tuning for {epochs} epochs (lr={lr})")
        
        target_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        
        optimizer = torch.optim.Adam(target_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        loss_fn = nn.CrossEntropyLoss()
        scaler = GradScaler(enabled=torch.cuda.is_available())
        
        best_composite_score = -1.0
        patience_counter = 0
        best_model_state = None
        
        # No fault injection during training - use standard fine-tuning

        for epoch in range(epochs):
            # Training loop
            target_model.train()
            for x, y in self.train_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad(set_to_none=True)

                # Standard forward pass
                with autocast(device_type=self.device.type, enabled=torch.cuda.is_available()):
                    outputs = target_model(x)
                    loss = loss_fn(outputs, y)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(target_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                # Enforce sparsity after each optimizer step
                with torch.no_grad():
                    for name, param in target_model.named_parameters():
                        if name in pruning_masks:
                            mask = pruning_masks[name]
                            if mask.shape != param.shape:
                                if mask.numel() == param.numel():
                                    mask = mask.view(param.shape)
                                else:
                                    continue
                            param.data.mul_(mask.to(param.device))
            
            # Validation with reliability check
            target_model.eval()
            
            # 1. Evaluate Accuracy
            correct, total = 0, 0
            with torch.inference_mode():
                for x_val, y_val in self.val_dataloader:
                    x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                    with autocast(device_type=self.device.type, enabled=torch.cuda.is_available()): # Mixed-precision for consistency
                        outputs = target_model(x_val)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_val.size(0)
                    correct += (predicted == y_val).sum().item()
            current_val_acc = 100 * correct / total
            
            # Skip reliability evaluation during fine-tuning - only use accuracy for early stopping
            # Reliability is evaluated once at the end by the GA evaluation system
            
            # Composite Score for early stopping (accuracy-based only)
            composite_score = current_val_acc
            
            print(f"  Epoch {epoch+1}: Acc={current_val_acc:.2f}%, Score={composite_score:.2f} (patience: {patience_counter})")

            # Early stopping logic based on composite score
            if composite_score > best_composite_score:
                best_composite_score = composite_score
                patience_counter = 0
                best_model_state = copy.deepcopy(target_model.state_dict())
                print(f"    -> NEW BEST MODEL (Score: {best_composite_score:.2f})")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_model_state is not None:
            target_model.load_state_dict(best_model_state)
            print(f"  Restored best model with composite score: {best_composite_score:.2f}")
        
        target_model.eval()
        cleanup_memory()
        
        return target_model


class GlobalPruningStrategyAgent:
    """Agent that generates pruning masks based on global sensitivity ranking."""
    
    def __init__(self, global_sparsity: float, model_state_dict: Dict, score_dir_path: str):
      
        self.global_sparsity = global_sparsity
        self.model_state_dict = model_state_dict
        self.score_dir_path = score_dir_path
        self.layer_score_files = get_layer_score_files_map(score_dir_path, model_state_dict)
        
        # Get prunable layers that have sensitivity scores - USE SENSITIVITY-BASED ORDERING
        available_layers = [
            name for name in model_state_dict.keys()
            if ('features' in name or 'classifier' in name) and 'weight' in name
            and name in self.layer_score_files
        ]

        # Use sensitivity-based ordering to match initialization
        self.prunable_layers = get_sensitivity_based_layer_ordering(available_layers, score_dir_path)
    
    def generate_pruning_mask(self, device: torch.device = torch.device('cpu')) -> Dict[str, torch.Tensor]:
       
        print(f"Generating global pruning masks (target sparsity: {self.global_sparsity:.1f}%)")
        
        # Step 1: Load all sensitivity scores into memory for performance
        print("  Loading all sensitivity scores into memory...")
        all_scores_list = []
        layer_scores_map = {}
        total_weights = 0
        
        for layer_name in self.prunable_layers:
            score_file_path = self.layer_score_files.get(layer_name)
            if not score_file_path:
                print(f"Warning: No score file for {layer_name}. Skipping.")
                continue
            try:
                scores_df = pd.read_csv(score_file_path)
                scores = scores_df['sensitivity_score'].astype(np.float32).values
                if self.model_state_dict[layer_name].numel() != len(scores):
                    print(f"Warning: Dimension mismatch for {layer_name}. Skipping.")
                    continue
                
                all_scores_list.append(scores)
                layer_scores_map[layer_name] = scores
                total_weights += len(scores)
            except Exception as e:
                print(f"Error loading scores for {layer_name}: {e}. Skipping.")
                continue
        
        if total_weights == 0:
            print("Error: No valid sensitivity scores found!")
            return {}
            
        print(f"  Loaded {total_weights} scores.")
        all_scores_flat = np.concatenate(all_scores_list)
        del all_scores_list

        # Step 2: Find the global pruning threshold using a fast O(n) partition
        num_to_prune = int(total_weights * (self.global_sparsity / 100.0))
        
        pruning_masks = {}
        if num_to_prune == 0:
            print("  No weights to prune.")
            for layer_name, scores in layer_scores_map.items():
                pruning_masks[layer_name] = torch.ones(len(scores), device=device).float()
            return pruning_masks

        print(f"  Finding global pruning threshold for {num_to_prune} weights...")
        
        # Fix for identical scores: Use argsort for precise control over ties
        sorted_indices = np.argsort(all_scores_flat)
        
        # Get the exact indices to prune (lowest sensitivity scores)
        indices_to_prune = sorted_indices[:num_to_prune]
        prune_set = set(indices_to_prune)
        
        # Calculate effective threshold for logging
        if num_to_prune > 0 and num_to_prune < len(sorted_indices):
            last_pruned_score = all_scores_flat[sorted_indices[num_to_prune - 1]]
            first_kept_score = all_scores_flat[sorted_indices[num_to_prune]]
            threshold = (last_pruned_score + first_kept_score) / 2.0
            print(f"  DEBUG: Pruning exactly {num_to_prune} weights using index-based selection")
            print(f"  DEBUG: last_pruned_score={last_pruned_score:.8f}, first_kept_score={first_kept_score:.8f}")
        else:
            threshold = all_scores_flat.min() - 1.0 if num_to_prune == 0 else all_scores_flat.max() + 1.0
        
        print(f"  DEBUG: effective_threshold={threshold:.8f}, will_prune={len(prune_set)} weights")
        
        # Store indices for precise mask generation
        self._global_prune_indices = prune_set
        self._total_weights_processed = 0
        
        del all_scores_flat

        # Step 3: Generate masks using precise index-based selection
        print("  Generating masks using index-based pruning selection...")
        for layer_name, scores in layer_scores_map.items():
            layer_size = len(scores)
            layer_start_idx = self._total_weights_processed
            layer_end_idx = self._total_weights_processed + layer_size
            
            # Find which indices in this layer should be pruned
            layer_prune_indices = []
            for global_idx in self._global_prune_indices:
                if layer_start_idx <= global_idx < layer_end_idx:
                    local_idx = global_idx - layer_start_idx
                    layer_prune_indices.append(local_idx)
            
            # Create mask: 1 = keep, 0 = prune
            mask = torch.ones(layer_size, device=device).float()
            if layer_prune_indices:
                mask[layer_prune_indices] = 0.0
            
            pruning_masks[layer_name] = mask
            weights_pruned = len(layer_prune_indices)
            layer_sparsity = (weights_pruned / layer_size) * 100
            print(f"    {layer_name}: {weights_pruned}/{layer_size} pruned ({layer_sparsity:.1f}%)")
            
            self._total_weights_processed += layer_size

        # Step 4: REMOVED - No layer protection, using precise index-based allocation
        print("  No layer protection - using precise index-based allocation")

        # Step 4: Verify global sparsity and log final stats
        total_pruned = sum((mask == 0).sum().item() for mask in pruning_masks.values())
        actual_global_sparsity = (total_pruned / total_weights) * 100 if total_weights > 0 else 0
        print(f"Achieved global sparsity: {actual_global_sparsity:.1f}% (target: {self.global_sparsity:.1f}%)")
        
        return pruning_masks
    
    def _apply_layer_protection(self, weights_to_prune: set, layer_boundaries: Dict) -> set:
        """Apply layer protection to prevent complete layer destruction."""
        protected_weights = set(weights_to_prune)
        min_survival_rate = 0.05  # At least 5% weights per layer must survive
        
        for layer_name, boundary in layer_boundaries.items():
            layer_size = boundary['count']
            min_survival = int(layer_size * min_survival_rate)
            
            # Count how many weights from this layer are marked for pruning
            layer_prune_count = sum(1 for (target_layer, _) in weights_to_prune 
                                  if target_layer == layer_name)
            
            # If we're pruning too much from this layer, protect some weights
            if layer_prune_count > layer_size - min_survival:
                excess_pruning = layer_prune_count - (layer_size - min_survival)
                
                # Find weights from this layer that are marked for pruning
                layer_pruned_items = [(target_layer, local_idx) for (target_layer, local_idx) in weights_to_prune 
                                    if target_layer == layer_name]
                
                # Load sensitivity scores to protect the most sensitive ones
                try:
                    score_file_path = self.layer_score_files.get(layer_name)
                    scores_df = pd.read_csv(score_file_path)
                    layer_scores = scores_df['sensitivity_score'].astype(float).values
                    
                    # Get scores for weights marked for pruning and protect most sensitive
                    scored_pruned = [(layer_scores[local_idx], (target_layer, local_idx)) 
                                   for (target_layer, local_idx) in layer_pruned_items]
                    scored_pruned.sort(key=lambda x: x[0], reverse=True)  # Most sensitive first
                    
                    # Remove the most sensitive weights from pruning list
                    for i in range(min(excess_pruning, len(scored_pruned))):
                        _, weight_to_protect = scored_pruned[i]
                        protected_weights.discard(weight_to_protect)
                    
                    print(f"  Layer protection: saved {min(excess_pruning, len(scored_pruned))} weights in {layer_name}")
                    
                except Exception as e:
                    print(f"Warning: Could not apply layer protection for {layer_name}: {e}")
        
        return protected_weights
    
    def get_projected_sparsity(self, original_total_params: int) -> float:
        """Calculate projected overall sparsity."""
        return min(self.global_sparsity, 99.9)


class ConstrainedLayerwisePruningAgent:
    """Layer-wise agent with global sparsity constraints for reliable benchmarking."""
    
    def __init__(self, strategy_params: List[float], target_global_sparsity: float,
                 model_state_dict: Dict, score_dir_path: str):
        self.strategy_params = strategy_params
        self.target_global_sparsity = target_global_sparsity
        self.model_state_dict = model_state_dict
        self.score_dir_path = score_dir_path
        self.layer_score_files = get_layer_score_files_map(score_dir_path, model_state_dict)
        
        # Get prunable layers that have sensitivity scores - USE SENSITIVITY-BASED ORDERING
        available_layers = [
            name for name in model_state_dict.keys()
            if ('features' in name or 'classifier' in name) and 'weight' in name
            and name in self.layer_score_files
        ]

        # Use sensitivity-based ordering to match initialization
        self.prunable_layers = get_sensitivity_based_layer_ordering(available_layers, score_dir_path)
        
        if len(strategy_params) != len(self.prunable_layers):
            raise ValueError(
                f"Strategy params length ({len(strategy_params)}) must match "
                f"prunable layers length ({len(self.prunable_layers)})"
            )
    
    def _calculate_projected_sparsity_unconstrained(self) -> float:
        """Calculate what sparsity would be without global constraint."""
        total_weights = 0
        total_pruned = 0
        
        for i, layer_name in enumerate(self.prunable_layers):
            layer_params = self.model_state_dict[layer_name].numel()
            # Allow 0.0% for complete layer protection in sequential approach
            effective_percentile = max(0.0, min(self.strategy_params[i], 99.9))
            layer_pruned = int(layer_params * (effective_percentile / 100.0))
            
            total_weights += layer_params
            total_pruned += layer_pruned
        
        return (total_pruned / total_weights) * 100.0 if total_weights > 0 else 0.0
    
    def _calculate_constraint_rescale_factor(self) -> float:
        """Calculate rescale factor to meet global sparsity target."""
        projected_sparsity = self._calculate_projected_sparsity_unconstrained()
        
        if projected_sparsity <= 0.1:
            return 1.0  # No rescaling needed for very low sparsity
        
        # Calculate rescale factor to hit target
        rescale_factor = self.target_global_sparsity / projected_sparsity
        
        # Clamp rescale factor to reasonable bounds
        rescale_factor = max(0.1, min(rescale_factor, 2.0))
        
        return rescale_factor
    
    def generate_pruning_mask(self, device: torch.device = torch.device('cpu')) -> Dict[str, torch.Tensor]:
        # Calculate constraint rescale factor
        rescale_factor = self._calculate_constraint_rescale_factor()
        projected_unconstrained = self._calculate_projected_sparsity_unconstrained()
        
        print(f"Constrained layer-wise pruning:")
        print(f"  Target global sparsity: {self.target_global_sparsity:.1f}%")
        print(f"  Unconstrained projection: {projected_unconstrained:.1f}%") 
        print(f"  Rescale factor: {rescale_factor:.3f}")
        
        # Apply rescaling to all layer parameters
        constrained_params = []
        for param in self.strategy_params:
            constrained_param = param * rescale_factor
            constrained_param = max(0.1, min(constrained_param, 99.9))  # Safety bounds
            constrained_params.append(constrained_param)
        
        print(f"  Original params: {[f'{p:.1f}%' for p in self.strategy_params]}")
        print(f"  Constrained params: {[f'{p:.1f}%' for p in constrained_params]}")
        
        pruning_masks = {}
        
        for i, layer_name in enumerate(self.prunable_layers):
            effective_percentile = constrained_params[i]
            
            weights = self.model_state_dict[layer_name]
            score_file_path = self.layer_score_files.get(layer_name)
            
            if not score_file_path:
                print(f"Warning: No score file for {layer_name}. Using improved sensitivity.")
                sensitivity_scores = self._compute_improved_sensitivity(layer_name, weights)
            else:
                try:
                    # Load sensitivity scores
                    scores_df = pd.read_csv(score_file_path)
                    raw_scores = scores_df['sensitivity_score'].astype(float).values
                    normalized_scores = self._normalize_sensitivity_scores(raw_scores)
                    print(f"DEBUG {layer_name}: raw_max={np.max(raw_scores):.2e}, normalized_max={np.max(normalized_scores):.6f}")
                    sensitivity_scores = torch.from_numpy(normalized_scores).to(device)

                    # Validate dimensions first
                    if weights.numel() != len(sensitivity_scores):
                        print(f"Warning: Dimension mismatch for {layer_name}. Using improved sensitivity.")
                        sensitivity_scores = self._compute_improved_sensitivity(layer_name, weights)
                    # Check if sensitivity scores are broken and use improved scoring if needed
                    elif self._scores_are_broken(sensitivity_scores, layer_name):
                        print(f"Computing improved sensitivity scores for {layer_name}")
                        sensitivity_scores = self._compute_improved_sensitivity(layer_name, weights)
                    else:
                        print(f"Using pre-computed sensitivity scores for {layer_name}")

                except Exception as e:
                    print(f"Error loading scores for {layer_name}: {e}")
                    print(f"Falling back to improved sensitivity computation")
                    sensitivity_scores = self._compute_improved_sensitivity(layer_name, weights)

            # Get weight magnitudes for vulnerability-based pruning (MENTOR'S APPROACH)
            weight_magnitudes = torch.abs(weights.view(-1)).to(device)

            # Generate mask using MENTOR'S VULNERABILITY-BASED APPROACH
            mask = self._create_sensitivity_based_mask(
                sensitivity_scores, effective_percentile, device, weight_magnitudes
            )
            pruning_masks[layer_name] = mask
            
            # Log pruning statistics
            num_pruned = (mask == 0).sum().item()
            total_weights = len(mask)
            actual_prune_percent = (num_pruned / total_weights) * 100 if total_weights > 0 else 0
            
            print(f"  {layer_name}: {effective_percentile:.1f}% -> "
                  f"{num_pruned}/{total_weights} pruned ({actual_prune_percent:.1f}%)")
        
        # Verify achieved global sparsity
        total_pruned = sum((mask == 0).sum().item() for mask in pruning_masks.values())
        total_weights = sum(mask.numel() for mask in pruning_masks.values())
        actual_global_sparsity = (total_pruned / total_weights) * 100 if total_weights > 0 else 0
        
        print(f"  Achieved global sparsity: {actual_global_sparsity:.1f}% (target: {self.target_global_sparsity:.1f}%)")
        
        return pruning_masks
    
    def _create_sensitivity_based_mask(self, sensitivity_scores: torch.Tensor, 
                                     percentile: float, device: torch.device) -> torch.Tensor:
        total_weights = len(sensitivity_scores)
        num_to_prune = int(total_weights * (percentile / 100.0))
        
        if num_to_prune <= 0:
            return torch.ones(total_weights, device=device).float()
        
        # Find indices of weights with lowest sensitivity scores
        indices_to_prune = torch.argsort(sensitivity_scores)[:num_to_prune]
        
        # Create mask
        mask = torch.ones(total_weights, device=device).float()
        mask[indices_to_prune] = 0.0
        
        return mask
    
    def get_projected_sparsity(self, original_total_params: int) -> float:
        """Return target global sparsity (always accurate due to constraints)."""
        return min(self.target_global_sparsity, 99.9)
    
    # Add the same improved sensitivity methods to ConstrainedLayerwisePruningAgent
    def _scores_are_broken(self, sensitivity_scores: torch.Tensor, layer_name: str) -> bool:
        """Detect if sensitivity scores are broken (overflow/underflow/all zeros)."""
        if len(sensitivity_scores) == 0:
            return True
            
        # IMPROVED: Only check for mathematical invalidity, not extreme values
        has_inf = torch.isinf(sensitivity_scores).any()
        has_nan = torch.isnan(sensitivity_scores).any()
        
        # REMOVED: has_overflow check - high values (1e37) indicate important weights!
        # REMOVED: all_zeros check - zero layers should get uniform low importance
        
        # IMPROVED: Only reject if ALL scores are identical (no ranking possible)
        unique_vals = torch.unique(sensitivity_scores)
        completely_uniform = len(unique_vals) == 1  # More permissive than <= 1
        
        is_broken = has_inf or has_nan or completely_uniform
        
        if is_broken:
            print(f"    Broken sensitivity detected: inf={has_inf}, nan={has_nan}, "
                  f"uniform={completely_uniform} (unique_vals={len(unique_vals)})")
        else:
            print(f"    Sensitivity scores accepted: {len(unique_vals)} unique values, "
                  f"max={sensitivity_scores.max().item():.2e}")
        
        return is_broken
    
    def _extract_layer_index(self, layer_name: str) -> int:
        """Extract layer index for importance weighting."""
        try:
            # Extract number from layer name like "features.0.weight" -> 0
            parts = layer_name.split('.')
            for part in parts:
                if part.isdigit():
                    return int(part)
        except:
            pass
        return 999  # Default high index if parsing fails
    
    def _compute_improved_sensitivity(self, layer_name: str, weights: torch.Tensor) -> torch.Tensor:
        """Compute improved sensitivity scores when original scores are broken."""
        device = weights.device
        weights_flat = weights.view(-1)
        
        # Method 1: Magnitude-based scoring (reliable baseline)
        magnitude_scores = torch.abs(weights_flat)
        
        # Method 2: Layer-type adaptive scoring
        layer_scores = self._get_layer_adaptive_scores(layer_name, weights_flat)
        
        # Method 3: Position-based importance (for conv layers)
        position_scores = self._get_position_importance(weights, layer_name)
        
        # Combine scores with proper normalization
        combined_scores = self._combine_and_normalize_scores(
            magnitude_scores, layer_scores, position_scores, layer_name
        )
        
        return combined_scores
    
    def _get_layer_adaptive_scores(self, layer_name: str, weights_flat: torch.Tensor) -> torch.Tensor:
        """Different scoring strategies based on layer type."""
        base_magnitude = torch.abs(weights_flat)
        
        if 'features' in layer_name:
            # Convolutional layers: Use magnitude with early-layer protection
            layer_idx = self._extract_layer_index(layer_name)
            
            # Early conv layers (0-2) are more critical
            if layer_idx <= 2:
                importance_multiplier = 2.0  # 2x importance for early layers
            elif layer_idx <= 6:
                importance_multiplier = 1.5  # 1.5x importance for middle layers  
            else:
                importance_multiplier = 1.0  # Normal importance for late layers
                
            return base_magnitude * importance_multiplier
            
        elif 'classifier' in layer_name:
            # Fully connected layers: Prevent all-zero issue with minimum floor
            min_score = base_magnitude.max() * 0.05  # 5% of max as minimum
            protected_scores = torch.maximum(base_magnitude, 
                                           torch.full_like(base_magnitude, min_score))
            
            # First classifier layer is most critical
            layer_idx = self._extract_layer_index(layer_name) 
            if layer_idx == 0:  # classifier.0 (features -> first FC)
                return protected_scores * 1.8  # Extra protection
            else:
                return protected_scores * 1.2  # Standard protection
                
        else:
            # Default case: pure magnitude
            return base_magnitude
    
    def _get_position_importance(self, weights: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Assign importance based on weight position (useful for conv layers)."""
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
    
    def _combine_and_normalize_scores(self, magnitude_scores: torch.Tensor, 
                                    layer_scores: torch.Tensor, 
                                    position_scores: torch.Tensor,
                                    layer_name: str) -> torch.Tensor:
        """Combine different scoring methods with proper normalization."""
        
        # Normalize each component to [0, 1] range using min-max scaling
        def normalize_scores(scores):
            min_val = scores.min()
            max_val = scores.max()
            if max_val > min_val:
                return (scores - min_val) / (max_val - min_val)
            else:
                return torch.ones_like(scores) * 0.5  # Fallback if all identical
        
        mag_norm = normalize_scores(magnitude_scores)
        layer_norm = normalize_scores(layer_scores)  
        pos_norm = normalize_scores(position_scores)
        
        # Layer-type specific combination weights
        if 'classifier' in layer_name:
            # Classifier layers: Emphasize magnitude and layer-specific scoring
            combined = 0.6 * mag_norm + 0.35 * layer_norm + 0.05 * pos_norm
        elif 'features' in layer_name:
            # Convolutional layers: Balanced combination with position importance
            combined = 0.4 * mag_norm + 0.4 * layer_norm + 0.2 * pos_norm
        else:
            # Default: Magnitude-focused
            combined = 0.7 * mag_norm + 0.2 * layer_norm + 0.1 * pos_norm
        
        # Final normalization and conversion to meaningful range
        final_scores = normalize_scores(combined)
        
        # Add small epsilon to prevent exact zeros (helps with sorting stability)
        epsilon = 1e-8
        final_scores = final_scores + epsilon
        
        return final_scores


class BalancedGlobalPruningStrategyAgent:
    """Global agent with layer-wise balance constraints to prevent extreme imbalances."""
    
    def __init__(self, global_sparsity: float, model_state_dict: Dict, 
                 score_dir_path: str, balance_factor: float = 20.0):
        
        self.global_sparsity = global_sparsity
        self.balance_factor = balance_factor  # Allowed variation around target (e.g., ±20%)
        self.model_state_dict = model_state_dict
        self.score_dir_path = score_dir_path
        self.layer_score_files = get_layer_score_files_map(score_dir_path, model_state_dict)
        
        # Get prunable layers that have sensitivity scores - USE SENSITIVITY-BASED ORDERING
        available_layers = [
            name for name in model_state_dict.keys()
            if ('features' in name or 'classifier' in name) and 'weight' in name
            and name in self.layer_score_files
        ]

        # Use sensitivity-based ordering to match initialization
        self.prunable_layers = get_sensitivity_based_layer_ordering(available_layers, score_dir_path)
        
        print(f"BalancedGlobalAgent: target={global_sparsity:.1f}%, balance=±{balance_factor:.1f}%")
    
    def generate_pruning_mask(self, device: torch.device = torch.device('cpu')) -> Dict[str, torch.Tensor]:
        
        print(f"Generating balanced global pruning masks (target: {self.global_sparsity:.1f}%)")
        
        # Calculate layer-wise sparsity bounds
        min_layer_sparsity = max(10.0, self.global_sparsity - self.balance_factor)
        max_layer_sparsity = min(95.0, self.global_sparsity + self.balance_factor)
        
        print(f"  Layer sparsity bounds: [{min_layer_sparsity:.1f}%, {max_layer_sparsity:.1f}%]")
        
        # Load sensitivity scores for all layers
        layer_scores_map = {}
        for layer_name in self.prunable_layers:
            score_file_path = self.layer_score_files.get(layer_name)
            if not score_file_path:
                print(f"Warning: No score file for {layer_name}. Skipping.")
                continue
            try:
                scores_df = pd.read_csv(score_file_path)
                scores = scores_df['sensitivity_score'].astype(np.float32).values
                if self.model_state_dict[layer_name].numel() != len(scores):
                    print(f"Warning: Dimension mismatch for {layer_name}. Skipping.")
                    continue
                layer_scores_map[layer_name] = scores
            except Exception as e:
                print(f"Error loading scores for {layer_name}: {e}. Skipping.")
                continue
        
        if not layer_scores_map:
            print("Error: No valid sensitivity scores found!")
            return {}
        
        # Generate masks using constrained layer-wise approach
        pruning_masks = {}
        total_pruned = 0
        total_weights = 0
        
        for layer_name, scores in layer_scores_map.items():
            layer_size = len(scores)
            total_weights += layer_size
            
            # Use global target as base, but allow controlled variation
            # Each layer gets the target sparsity, but GA can evolve the balance_factor
            target_layer_sparsity = self.global_sparsity
            
            # Clamp to safe bounds
            effective_sparsity = max(min_layer_sparsity, 
                                   min(target_layer_sparsity, max_layer_sparsity))
            
            # Calculate number of weights to prune in this layer
            num_to_prune = int(layer_size * (effective_sparsity / 100.0))
            
            if num_to_prune <= 0:
                mask = torch.ones(layer_size, device=device).float()
            else:
                # Use sensitivity-based ranking within this layer
                scores_tensor = torch.from_numpy(scores).to(device)
                indices_to_prune = torch.argsort(scores_tensor)[:num_to_prune]
                
                # Create mask
                mask = torch.ones(layer_size, device=device).float()
                mask[indices_to_prune] = 0.0
            
            pruning_masks[layer_name] = mask
            
            # Log layer results
            actual_pruned = (mask == 0).sum().item()
            actual_sparsity = (actual_pruned / layer_size) * 100 if layer_size > 0 else 0
            total_pruned += actual_pruned
            
            print(f"    {layer_name}: {actual_pruned}/{layer_size} pruned ({actual_sparsity:.1f}%)")
        
        # Verify overall sparsity
        actual_global_sparsity = (total_pruned / total_weights) * 100 if total_weights > 0 else 0
        print(f"  Achieved global sparsity: {actual_global_sparsity:.1f}% (target: {self.global_sparsity:.1f}%)")
        
        return pruning_masks
    
    def get_projected_sparsity(self, original_total_params: int) -> float:
        """Calculate projected overall sparsity."""
        return min(self.global_sparsity, 99.9)


class ArchitecturalPatternPruningAgent(PruningStrategyAgent):
    """
    Architectural pattern-based agent that reduces 19D space to 4D patterns:
    [base_sparsity, conv_bias, fc_bias, protection_level] -> generates 19 layer params
    """
    
    def __init__(self, pattern_params: List[float], model_state_dict: Dict, score_dir_path: str):
        # Validate pattern parameters
        if len(pattern_params) != 4:
            raise ValueError(f"ArchitecturalPatternPruningAgent requires exactly 4 parameters, got {len(pattern_params)}")
        
        # Store score_dir_path first for _generate_layer_parameters
        self.score_dir_path = score_dir_path

        # Extract pattern parameters with bounds
        self.base_sparsity = max(25.0, min(80.0, pattern_params[0]))
        self.conv_bias = max(-25.0, min(25.0, pattern_params[1]))
        self.fc_bias = max(-25.0, min(25.0, pattern_params[2]))
        self.protection_level = max(0.0, min(100.0, pattern_params[3]))

        # Generate 19 layer-specific parameters from 4 pattern parameters
        generated_layer_params = self._generate_layer_parameters(model_state_dict)

        # Initialize parent with generated parameters
        super().__init__(generated_layer_params, model_state_dict, score_dir_path)
        
        print(f"Pattern Agent: base={self.base_sparsity:.1f}%, conv_bias={self.conv_bias:+.1f}%, "
              f"fc_bias={self.fc_bias:+.1f}%, protection={self.protection_level:.1f}%")
    
    def _generate_layer_parameters(self, model_state_dict: Dict) -> List[float]:
        """Generate 19 layer parameters from 4 pattern parameters."""
        
        # Get all prunable layers in consistent order - USE SENSITIVITY-BASED ORDERING
        available_layers = [name for name in model_state_dict.keys()
                           if ('features' in name or 'classifier' in name) and 'weight' in name]

        # Use sensitivity-based ordering to match initialization
        prunable_layers = get_sensitivity_based_layer_ordering(available_layers, self.score_dir_path)
        
        layer_params = []
        
        for layer_name in prunable_layers:
            # Classify layer architecture
            arch_type, base_protection = self._classify_layer_architecture(layer_name)
            
            # Start with base sparsity
            target_sparsity = self.base_sparsity
            
            # Apply architectural bias
            if arch_type.startswith('conv'):
                target_sparsity += self.conv_bias
            else:  # FC layers
                target_sparsity += self.fc_bias
            
            # Apply protection based on architectural importance
            protection_factor = (base_protection / 100.0) * (self.protection_level / 100.0)
            protection_reduction = target_sparsity * protection_factor * 0.4  # Max 40% reduction
            
            final_sparsity = target_sparsity - protection_reduction
            
            # Apply architectural bounds
            final_sparsity = self._apply_architectural_bounds(arch_type, final_sparsity)
            
            layer_params.append(final_sparsity)
        
        return layer_params
    
    def _classify_layer_architecture(self, layer_name: str) -> Tuple[str, float]:
        """Classify layer architecture and assign base protection level."""
        
        if layer_name in ['features.0.weight', 'features.4.weight']:
            return 'conv_early', 90.0  # Very protective of early layers
        elif layer_name in ['features.8.weight', 'features.11.weight', 'features.15.weight']:
            return 'conv_mid', 70.0    # Moderate protection
        elif 'features' in layer_name:
            return 'conv_late', 50.0   # Less protection
        elif layer_name == 'classifier.0.weight':
            return 'fc_early', 75.0    # First FC important
        elif layer_name == 'classifier.3.weight':
            return 'fc_mid', 40.0      # Middle FC often redundant
        elif layer_name == 'classifier.6.weight':
            return 'fc_final', 60.0    # Final layer moderately important
        else:
            return 'other', 60.0       # Default
    
    def _apply_architectural_bounds(self, arch_type: str, sparsity: float) -> float:
        """Apply architectural-specific bounds to prevent extreme pruning."""
        
        if arch_type == 'conv_early':
            return max(2.0, min(20.0, sparsity))   # Very conservative
        elif arch_type == 'conv_mid':
            return max(5.0, min(50.0, sparsity))   # Moderate
        elif arch_type == 'conv_late':
            return max(10.0, min(70.0, sparsity))  # More aggressive
        elif arch_type == 'fc_early':
            return max(8.0, min(60.0, sparsity))   # Moderate for first FC
        elif arch_type == 'fc_mid':
            return max(20.0, min(85.0, sparsity))  # Can be aggressive
        elif arch_type == 'fc_final':
            return max(15.0, min(75.0, sparsity))  # Moderate for output
        else:
            return max(5.0, min(70.0, sparsity))   # Default bounds


class SensitivityAwarePruningStrategyAgent(PruningStrategyAgent):
    """Enhanced agent that uses actual sensitivity scores from CSV files for weight-level pruning decisions."""
    
    def __init__(self, strategy_params: List[float], model_state_dict: Dict, 
                 score_dir_path: str = "VGG11/vgg_weight_grades/vgg_weight_sensitivity_score/"):
        super().__init__(strategy_params, model_state_dict, score_dir_path)
        self.score_dir_path = score_dir_path
        
    def _load_actual_sensitivity_scores(self, layer_name: str, expected_size: int) -> Optional[np.ndarray]:
        """Load actual sensitivity scores from CSV files."""
        import glob
        
        pattern = os.path.join(self.score_dir_path, f"weight_sensitivity_scores_{layer_name}.csv")
        files = glob.glob(pattern)
        
        if files:
            try:
                df = pd.read_csv(files[0])
                if 'sensitivity_score' in df.columns:
                    raw_scores = df['sensitivity_score'].astype(float).values
                    
                    # Validate size matches expected weight count
                    if len(raw_scores) != expected_size:
                        print(f"    Warning: Sensitivity score count ({len(raw_scores)}) doesn't match weight count ({expected_size}) for {layer_name}")
                        return None
                    
                    # Apply robust normalization for extreme values
                    normalized_scores = self._normalize_sensitivity_scores(raw_scores, layer_name)
                    return normalized_scores
                    
            except Exception as e:
                print(f"    Warning: Could not load sensitivity scores for {layer_name}: {e}")
        
        return None
    
    def _normalize_sensitivity_scores(self, scores: np.ndarray, layer_name: str = "unknown") -> np.ndarray:
        """Normalize sensitivity scores preserving natural layer characteristics."""
        finite_mask = np.isfinite(scores)
        if not np.any(finite_mask):
            return np.ones_like(scores) * 0.5

        finite_scores = scores[finite_mask]
        abs_scores = np.abs(finite_scores)

        # Check if scores are essentially uniform (all zeros or identical values)
        unique_values = len(np.unique(abs_scores))
        score_std = np.std(abs_scores)

        if unique_values <= 2 or score_std < 1e-10:
            # Layer has no meaningful sensitivity variation - assign low importance
            if layer_name != "unknown":
                print(f"    {layer_name}: No sensitivity variation detected - assigning uniform low scores")
            return np.ones_like(scores) * 0.25  # Low uniform sensitivity
        
        if np.max(abs_scores) > 1e30:  # Handle extreme overflow cases like 1e37
            # Use robust log-scale normalization that PRESERVES natural distribution shape
            log_scores = np.log10(abs_scores + 1e-15)
            
            # Simple min-max on log scale - preserves relative differences
            log_min, log_max = np.min(log_scores), np.max(log_scores)
            if log_max > log_min:
                normalized = (log_scores - log_min) / (log_max - log_min)
            else:
                normalized = np.ones_like(log_scores) * 0.5
                
        else:
            # Standard min-max normalization for reasonable values
            min_val, max_val = np.min(abs_scores), np.max(abs_scores)
            if max_val > min_val:
                normalized = (abs_scores - min_val) / (max_val - min_val)
            else:
                normalized = np.ones_like(abs_scores) * 0.5
        
        # Fill result array
        result = np.ones_like(scores) * 0.5
        result[finite_mask] = normalized

        # Calculate and report actual characteristics
        if layer_name != "unknown":
            sensitive_ratio = np.sum(result > 0.5) / len(result)
            score_range = (np.min(result), np.max(result))
            print(f"    {layer_name}: sensitive_ratio={sensitive_ratio:.3f}, range={score_range[0]:.3f}-{score_range[1]:.3f}")

        return result
    
    def _compute_improved_sensitivity(self, layer_name: str, weights: torch.Tensor) -> torch.Tensor:
        """Compute sensitivity scores using actual CSV data when available, fallback to computed scores."""
        device = weights.device
        weights_flat = weights.view(-1)
        
        # First try to load actual sensitivity scores from CSV files
        actual_sensitivity_scores = self._load_actual_sensitivity_scores(layer_name, weights_flat.shape[0])
        
        if actual_sensitivity_scores is not None:
            print(f"    Using actual sensitivity scores for {layer_name} (from CSV)")
            return torch.tensor(actual_sensitivity_scores, device=device, dtype=torch.float32)
        
        # Fallback to parent's computed scores when CSV not available
        print(f"    Using computed sensitivity scores for {layer_name} (CSV not available)")
        return super()._compute_improved_sensitivity(layer_name, weights)