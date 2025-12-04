"""Pattern learning system that builds on sensitivity-informed initialization."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ReliabilityPattern:
    """A learned pattern that correlates with high reliability."""
    pattern_id: str
    layer_ranges: Dict[str, Tuple[float, float]]  # layer_name -> (min_pruning, max_pruning)
    reliability_stats: Dict[str, float]  # mean, std, min, max reliability
    sample_count: int
    discovery_generation: int
    confidence_score: float
    

class PatternLearner:
    """Learns reliability patterns from sensitivity-informed strategies."""
    
    def __init__(self, sensitivity_data: Dict[str, Dict]):
        self.sensitivity_data = sensitivity_data
        self.learned_patterns: Dict[str, ReliabilityPattern] = {}
        self.individual_history: List[Dict] = []
        self.generation = 0
        
    def record_individual(self, individual: List[float], metrics: Dict, generation: int):
        """Record individual and its performance for pattern learning."""
        
        record = {
            'individual': individual.copy(),
            'accuracy': metrics.get('accuracy', 0),
            'reliability': metrics.get('reliability', 0),
            'sparsity': metrics.get('sparsity', 0),
            'generation': generation
        }
        
        self.individual_history.append(record)
        
        # Every 5 generations, analyze patterns
        if generation > 0 and generation % 5 == 0:
            self._analyze_and_update_patterns()
    
    def _analyze_and_update_patterns(self):
        """Analyze recent individuals to discover reliability patterns."""
        
        if len(self.individual_history) < 10:  # Need minimum data
            return
        
        print(f"Analyzing patterns at generation {self.generation}...")
        
        # Get high-reliability individuals from recent generations
        recent_history = [r for r in self.individual_history if r['generation'] >= self.generation - 10]
        
        # Sort by reliability
        recent_history.sort(key=lambda x: x['reliability'], reverse=True)
        
        # Top 20% as high-reliability exemplars
        top_count = max(3, len(recent_history) // 5)
        high_reliability_individuals = recent_history[:top_count]
        
        if not high_reliability_individuals:
            return
        
        # Extract pattern from high-reliability individuals
        avg_reliability = np.mean([ind['reliability'] for ind in high_reliability_individuals])
        
        if avg_reliability > 75.0:  # Only learn from genuinely good patterns
            pattern = self._extract_pattern_from_individuals(high_reliability_individuals)
            if pattern:
                self.learned_patterns[pattern.pattern_id] = pattern
                print(f"  Learned new pattern '{pattern.pattern_id}' with reliability {avg_reliability:.1f}%")
    
    def _extract_pattern_from_individuals(self, individuals: List[Dict]) -> Optional[ReliabilityPattern]:
        """Extract common patterns from high-performing individuals."""
        
        if not individuals:
            return None
        
        layer_names = [f"layer_{i}" for i in range(len(individuals[0]['individual']))]  # Placeholder
        
        # Calculate ranges for each layer across high-performing individuals
        layer_ranges = {}
        
        for layer_idx, layer_name in enumerate(layer_names):
            values = [ind['individual'][layer_idx] for ind in individuals]
            
            if len(values) > 1:
                min_val = np.min(values)
                max_val = np.max(values)
                
                # Expand range slightly for generation flexibility
                range_expansion = (max_val - min_val) * 0.2
                expanded_min = max(5.0, min_val - range_expansion)
                expanded_max = min(95.0, max_val + range_expansion)
                
                layer_ranges[layer_name] = (expanded_min, expanded_max)
            else:
                # Single individual - create small range around it
                val = values[0]
                layer_ranges[layer_name] = (max(5.0, val - 5), min(95.0, val + 5))
        
        # Pattern statistics
        reliabilities = [ind['reliability'] for ind in individuals]
        reliability_stats = {
            'mean': np.mean(reliabilities),
            'std': np.std(reliabilities),
            'min': np.min(reliabilities),
            'max': np.max(reliabilities)
        }
        
        pattern = ReliabilityPattern(
            pattern_id=f"pattern_gen_{self.generation}_{len(self.learned_patterns)}",
            layer_ranges=layer_ranges,
            reliability_stats=reliability_stats,
            sample_count=len(individuals),
            discovery_generation=self.generation,
            confidence_score=min(1.0, len(individuals) / 10.0)  # Confidence based on sample size
        )
        
        return pattern
    
    def suggest_pattern_guided_individual(self) -> Optional[List[float]]:
        """Generate new individual based on learned patterns."""
        
        if not self.learned_patterns:
            return None
        
        # Weight patterns by reliability and confidence
        pattern_weights = {}
        for pattern_id, pattern in self.learned_patterns.items():
            weight = pattern.reliability_stats['mean'] * pattern.confidence_score
            pattern_weights[pattern_id] = weight
        
        # Select pattern probabilistically
        total_weight = sum(pattern_weights.values())
        if total_weight == 0:
            return None
        
        # Weighted random selection
        rand_val = np.random.random() * total_weight
        cumulative = 0
        selected_pattern = None
        
        for pattern_id, weight in pattern_weights.items():
            cumulative += weight
            if rand_val <= cumulative:
                selected_pattern = self.learned_patterns[pattern_id]
                break
        
        if not selected_pattern:
            return None
        
        # Generate individual within pattern ranges
        individual = []
        for layer_name, (min_val, max_val) in selected_pattern.layer_ranges.items():
            value = np.random.uniform(min_val, max_val)
            individual.append(value)
        
        return individual
    
    def get_exploration_ratio(self, generation: int, current_diversity: float) -> float:
        """Determine how much exploration vs pattern-guided generation to use."""
        
        base_exploration = 0.4  # Always maintain 40% exploration
        
        # Reduce exploration as we learn more patterns
        pattern_bonus = len(self.learned_patterns) * 0.05  # Up to -25% exploration
        
        # Increase exploration if diversity is low
        diversity_bonus = max(0, (0.3 - current_diversity) * 0.3)  # Up to +9% exploration
        
        final_ratio = base_exploration - pattern_bonus + diversity_bonus
        return max(0.2, min(0.8, final_ratio))  # Keep between 20-80%
    
    def update_generation(self, generation: int):
        """Update current generation for pattern learning."""
        self.generation = generation