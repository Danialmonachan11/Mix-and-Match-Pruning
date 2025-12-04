"""Robust sensitivity analyzer that handles overflow and provides meaningful pruning guidance."""

import pandas as pd
import numpy as np
import torch
import os
import glob
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LayerAnalysis:
    """Analysis results for a single layer."""
    layer_name: str
    total_weights: int
    architectural_type: str  # 'early_conv', 'mid_conv', 'late_conv', 'early_fc', 'mid_fc', 'final_fc'
    protection_level: str   # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    recommended_min_pruning: float  # Minimum safe pruning percentage
    recommended_max_pruning: float  # Maximum safe pruning percentage
    sensitivity_status: str  # 'VALID_SCORES', 'OVERFLOW_DETECTED', 'MISSING_SCORES', 'HEURISTIC_ONLY'
    actual_sensitivity_ratio: Optional[float]  # If scores are valid


class RobustSensitivityAnalyzer:
    """
    Robust sensitivity analyzer that:
    1. Detects and handles numerical overflow in sensitivity scores
    2. Provides architectural-aware pruning guidance
    3. Falls back gracefully to heuristics when scores are unusable
    4. Gives clear actionable recommendations for GA
    """
    
    def __init__(self, score_dir_path: str, model_state_dict: Dict[str, torch.Tensor]):
        self.score_dir_path = score_dir_path
        self.model_state_dict = model_state_dict
        self.layer_analyses: Dict[str, LayerAnalysis] = {}
        
        # VGG-11 architectural knowledge
        self.layer_architecture_map = self._create_architecture_map()
        
        print("🔍 Initializing Robust Sensitivity Analyzer...")
        self._analyze_all_layers()
    
    def _create_architecture_map(self) -> Dict[str, dict]:
        """Create architectural classification for VGG-11 layers."""
        return {
            # Early convolutional layers (most critical for feature extraction)
            'features.0.weight': {'type': 'early_conv', 'importance': 'CRITICAL', 'description': 'First conv layer'},
            'features.4.weight': {'type': 'early_conv', 'importance': 'CRITICAL', 'description': 'Second conv layer'},
            
            # Middle convolutional layers
            'features.8.weight': {'type': 'mid_conv', 'importance': 'HIGH', 'description': 'Third conv layer'},
            'features.11.weight': {'type': 'mid_conv', 'importance': 'HIGH', 'description': 'Fourth conv layer'},
            'features.15.weight': {'type': 'mid_conv', 'importance': 'HIGH', 'description': 'Fifth conv layer'},
            
            # Late convolutional layers (can handle more pruning)
            'features.18.weight': {'type': 'late_conv', 'importance': 'MEDIUM', 'description': 'Sixth conv layer'},
            'features.22.weight': {'type': 'late_conv', 'importance': 'MEDIUM', 'description': 'Seventh conv layer'},
            
            # Fully connected layers (often have many redundant weights)
            'classifier.0.weight': {'type': 'early_fc', 'importance': 'HIGH', 'description': 'First FC layer (feature->hidden)'},
            'classifier.3.weight': {'type': 'mid_fc', 'importance': 'MEDIUM', 'description': 'Second FC layer (hidden->hidden)'},
            'classifier.6.weight': {'type': 'final_fc', 'importance': 'LOW', 'description': 'Final FC layer (hidden->classes)'}
        }
    
    def _analyze_all_layers(self):
        """Analyze all layers and create actionable pruning guidance."""
        print("🔍 Analyzing layer sensitivity and architectural importance...")
        
        # Get all prunable layers from model
        prunable_layers = [name for name in self.model_state_dict.keys() 
                          if ('features' in name or 'classifier' in name) and 'weight' in name]
        
        for layer_name in prunable_layers:
            analysis = self._analyze_single_layer(layer_name)
            self.layer_analyses[layer_name] = analysis
            
            print(f"  📊 {layer_name}: {analysis.sensitivity_status} | "
                  f"{analysis.protection_level} | "
                  f"Safe range: {analysis.recommended_min_pruning:.1f}%-{analysis.recommended_max_pruning:.1f}%")
    
    def _analyze_single_layer(self, layer_name: str) -> LayerAnalysis:
        """Analyze a single layer comprehensively."""
        
        # Get basic layer info
        weights = self.model_state_dict[layer_name]
        total_weights = weights.numel()
        
        # Get architectural classification
        arch_info = self.layer_architecture_map.get(layer_name, {
            'type': 'unknown', 'importance': 'MEDIUM', 'description': 'Unknown layer'
        })
        
        # Try to load and validate sensitivity scores
        sensitivity_status, actual_sensitivity_ratio = self._load_and_validate_scores(layer_name)
        
        # Determine protection level and safe ranges
        protection_level, min_pruning, max_pruning = self._determine_protection_strategy(
            layer_name, arch_info, sensitivity_status, actual_sensitivity_ratio, total_weights
        )
        
        return LayerAnalysis(
            layer_name=layer_name,
            total_weights=total_weights,
            architectural_type=arch_info['type'],
            protection_level=protection_level,
            recommended_min_pruning=min_pruning,
            recommended_max_pruning=max_pruning,
            sensitivity_status=sensitivity_status,
            actual_sensitivity_ratio=actual_sensitivity_ratio
        )
    
    def _load_and_validate_scores(self, layer_name: str) -> Tuple[str, Optional[float]]:
        """Load sensitivity scores and detect problems."""
        
        # Try to find score file
        pattern = os.path.join(self.score_dir_path, f"weight_sensitivity_scores_{layer_name}.csv")
        files = glob.glob(pattern)
        
        if not files:
            return "MISSING_SCORES", None
        
        try:
            df = pd.read_csv(files[0])
            if 'sensitivity_score' not in df.columns:
                return "MISSING_SCORES", None
            
            raw_scores = df['sensitivity_score'].astype(float).values
            
            # Comprehensive score validation
            validation_result = self._validate_score_quality(raw_scores, layer_name)
            
            if validation_result['is_valid']:
                return "VALID_SCORES", validation_result['sensitivity_ratio']
            else:
                print(f"    ⚠️ {layer_name}: {validation_result['problem']} - using architectural heuristics")
                return "OVERFLOW_DETECTED", None
                
        except Exception as e:
            print(f"    ❌ {layer_name}: Error loading scores: {e}")
            return "MISSING_SCORES", None
    
    def _validate_score_quality(self, scores: np.ndarray, layer_name: str) -> Dict:
        """Comprehensively validate sensitivity score quality."""
        
        if len(scores) == 0:
            return {'is_valid': False, 'problem': 'Empty scores array'}
        
        # Check for overflow (your main problem!)
        max_abs_score = np.max(np.abs(scores))
        if max_abs_score > 1e30:  # Clear overflow threshold
            return {
                'is_valid': False, 
                'problem': f'Numerical overflow detected (max={max_abs_score:.2e})'
            }
        
        # Check for inf/nan
        if not np.all(np.isfinite(scores)):
            inf_count = np.sum(np.isinf(scores))
            nan_count = np.sum(np.isnan(scores))
            return {
                'is_valid': False,
                'problem': f'Non-finite values: {inf_count} inf, {nan_count} nan'
            }
        
        # Check for all-zero or all-identical
        unique_scores = np.unique(scores)
        if len(unique_scores) <= 2:  # Allow small variations due to float precision
            return {
                'is_valid': False,
                'problem': f'Insufficient variation: only {len(unique_scores)} unique values'
            }
        
        # Check for reasonable distribution
        if np.std(scores) < 1e-10:
            return {
                'is_valid': False,
                'problem': 'Scores have no meaningful variation (std < 1e-10)'
            }
        
        # Scores appear valid - calculate meaningful sensitivity ratio
        # Use a robust threshold based on score distribution
        threshold = np.percentile(np.abs(scores), 75)  # 75th percentile as sensitivity threshold
        sensitive_count = np.sum(np.abs(scores) > threshold)
        sensitivity_ratio = sensitive_count / len(scores)
        
        return {
            'is_valid': True,
            'sensitivity_ratio': sensitivity_ratio,
            'problem': None
        }
    
    def _determine_protection_strategy(self, layer_name: str, arch_info: dict, 
                                     sensitivity_status: str, actual_sensitivity_ratio: Optional[float],
                                     total_weights: int) -> Tuple[str, float, float]:
        """Determine protection level and safe pruning ranges."""
        
        # Start with architectural baseline
        arch_importance = arch_info['importance']
        arch_type = arch_info['type']
        
        # Adjust based on sensitivity scores (if valid)
        if sensitivity_status == "VALID_SCORES" and actual_sensitivity_ratio is not None:
            # Use actual sensitivity data to refine protection
            if actual_sensitivity_ratio > 0.8:  # Very sensitive
                protection_level = "CRITICAL"
            elif actual_sensitivity_ratio > 0.6:  # Moderately sensitive
                protection_level = "HIGH" 
            elif actual_sensitivity_ratio > 0.3:  # Some sensitivity
                protection_level = "MEDIUM"
            else:  # Low sensitivity
                protection_level = "LOW"
        else:
            # Fall back to architectural importance
            protection_level = arch_importance
        
        # Determine safe pruning ranges based on protection level and architecture
        if protection_level == "CRITICAL":
            if arch_type in ['early_conv']:
                min_pruning, max_pruning = 1.0, 10.0  # Very conservative for early conv
            elif arch_type in ['mid_conv']:
                min_pruning, max_pruning = 2.0, 15.0  # Conservative for mid conv
            else:
                min_pruning, max_pruning = 5.0, 25.0  # Conservative for FC
                
        elif protection_level == "HIGH":
            if arch_type in ['early_conv', 'mid_conv']:
                min_pruning, max_pruning = 5.0, 30.0  # Moderate for conv layers
            else:
                min_pruning, max_pruning = 10.0, 50.0  # More aggressive for FC
                
        elif protection_level == "MEDIUM":
            if arch_type in ['late_conv']:
                min_pruning, max_pruning = 15.0, 60.0  # Can prune late conv more
            elif arch_type in ['mid_fc']:
                min_pruning, max_pruning = 20.0, 70.0  # FC layers can handle more
            else:
                min_pruning, max_pruning = 10.0, 50.0  # Default moderate
                
        else:  # LOW protection
            if arch_type == 'final_fc':
                min_pruning, max_pruning = 30.0, 85.0  # Very aggressive for final layer
            else:
                min_pruning, max_pruning = 25.0, 75.0  # Aggressive but not extreme
        
        # Size-based adjustments (larger layers can typically handle more pruning)
        if total_weights > 10_000_000:  # Very large layers (like classifier.3)
            max_pruning = min(max_pruning + 15.0, 90.0)  # Can prune more
        elif total_weights < 1000:  # Very small layers
            max_pruning = max(max_pruning - 10.0, min_pruning + 5.0)  # Be more careful
        
        return protection_level, min_pruning, max_pruning
    
    def get_layer_guidance(self, layer_name: str) -> Optional[LayerAnalysis]:
        """Get guidance for a specific layer."""
        return self.layer_analyses.get(layer_name)
    
    def get_constrained_pruning_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get safe pruning ranges for all layers (for GA constraints)."""
        return {
            layer_name: (analysis.recommended_min_pruning, analysis.recommended_max_pruning)
            for layer_name, analysis in self.layer_analyses.items()
        }
    
    def print_comprehensive_analysis(self):
        """Print detailed analysis of all layers."""
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE SENSITIVITY & ARCHITECTURAL ANALYSIS")
        print("="*80)
        
        for layer_name, analysis in self.layer_analyses.items():
            print(f"\n📊 {layer_name}")
            print(f"   Architecture: {analysis.architectural_type} ({analysis.total_weights:,} weights)")
            print(f"   Sensitivity: {analysis.sensitivity_status}")
            if analysis.actual_sensitivity_ratio is not None:
                print(f"   Sensitivity Ratio: {analysis.actual_sensitivity_ratio:.2f}")
            print(f"   Protection Level: {analysis.protection_level}")
            print(f"   Safe Pruning Range: {analysis.recommended_min_pruning:.1f}% - {analysis.recommended_max_pruning:.1f}%")
        
        # Summary statistics
        valid_scores = sum(1 for a in self.layer_analyses.values() if a.sensitivity_status == "VALID_SCORES")
        overflow_detected = sum(1 for a in self.layer_analyses.values() if a.sensitivity_status == "OVERFLOW_DETECTED")
        missing_scores = sum(1 for a in self.layer_analyses.values() if a.sensitivity_status == "MISSING_SCORES")
        
        print(f"\n📈 SUMMARY:")
        print(f"   Valid sensitivity scores: {valid_scores}/{len(self.layer_analyses)}")
        print(f"   Overflow detected: {overflow_detected}/{len(self.layer_analyses)}")
        print(f"   Missing scores: {missing_scores}/{len(self.layer_analyses)}")
        
        if overflow_detected > 0:
            print(f"\n⚠️  WARNING: {overflow_detected} layers have overflow in sensitivity scores!")
            print(f"   Using architectural heuristics for these layers.")
        
        print("="*80)