"""Fault-aware population initialization for reliability-focused GA."""
import random
from typing import List, Dict


class FaultAwareInitializer:
    """
    Initialize GA population based on layer fault-impact scores.

    Key idea: Layers with high fault-impact should be pruned conservatively,
    while layers with low fault-impact can be pruned more aggressively.
    """

    def __init__(self, fault_impacts: Dict[str, float]):
        """
        Initialize fault-aware initializer.

        Args:
            fault_impacts: Dictionary mapping layer_name → fault_impact_score
                          Higher score = more critical under faults
        """
        self.fault_impacts = fault_impacts

        # Classify layers into tiers based on fault impact
        impacts = list(fault_impacts.values())
        if len(impacts) >= 3:
            # Sort and find thresholds (top 33% = high, middle 33% = medium, bottom 33% = low)
            sorted_impacts = sorted(impacts, reverse=True)
            self.high_threshold = sorted_impacts[len(impacts) // 3]
            self.medium_threshold = sorted_impacts[2 * len(impacts) // 3]
        else:
            # Fallback for small number of layers
            max_impact = max(impacts) if impacts else 1.0
            self.high_threshold = max_impact * 0.7
            self.medium_threshold = max_impact * 0.3

    def create_individual(self, prunable_layers: List[str],
                         min_percentile: float = 10.0,
                         max_percentile: float = 80.0) -> List[float]:
        """
        Create one individual with fault-aware percentiles.

        Args:
            prunable_layers: List of layer names
            min_percentile: Minimum pruning percentage
            max_percentile: Maximum pruning percentage

        Returns:
            List of pruning percentiles (one per layer)
        """
        individual = []

        for layer in prunable_layers:
            impact = self.fault_impacts.get(layer, 0.0)

            # Adapt pruning range based on fault impact
            if impact >= self.high_threshold:
                # CRITICAL LAYER: Conservative pruning (10-30%)
                layer_min = max(min_percentile, 10.0)
                layer_max = min(max_percentile, 30.0)
            elif impact >= self.medium_threshold:
                # MEDIUM IMPACT: Moderate pruning (20-50%)
                layer_min = max(min_percentile, 20.0)
                layer_max = min(max_percentile, 50.0)
            else:
                # LOW IMPACT: Aggressive pruning (40-80%)
                layer_min = max(min_percentile, 40.0)
                layer_max = min(max_percentile, 80.0)

            percentile = random.uniform(layer_min, layer_max)
            individual.append(percentile)

        return individual

    def create_population(self, prunable_layers: List[str],
                         pop_size: int,
                         min_percentile: float = 10.0,
                         max_percentile: float = 80.0,
                         fault_aware_ratio: float = 0.7) -> List[List[float]]:
        """
        Create entire population with fault-awareness.

        Args:
            prunable_layers: List of layer names
            pop_size: Population size
            min_percentile: Minimum pruning percentage
            max_percentile: Maximum pruning percentage
            fault_aware_ratio: Fraction of population to use fault-aware init (rest is random)

        Returns:
            Population of individuals
        """
        population = []

        # Calculate how many individuals use fault-aware vs random
        fault_aware_count = int(pop_size * fault_aware_ratio)

        for i in range(pop_size):
            if i < fault_aware_count:
                # Fault-aware individual
                individual = self.create_individual(prunable_layers, min_percentile, max_percentile)
            else:
                # Random exploration individual (no fault-awareness)
                individual = [random.uniform(min_percentile, max_percentile)
                             for _ in prunable_layers]

            population.append(individual)

        return population

    def get_layer_classification(self, layer_name: str) -> str:
        """Get classification string for a layer based on fault impact."""
        impact = self.fault_impacts.get(layer_name, 0.0)

        if impact >= self.high_threshold:
            return "CRITICAL"
        elif impact >= self.medium_threshold:
            return "MEDIUM"
        else:
            return "LOW"

    def print_layer_classifications(self, prunable_layers: List[str]):
        """Print fault-impact classification for all layers."""
        print(f"\n{'='*70}")
        print("FAULT-AWARE LAYER CLASSIFICATIONS")
        print(f"{'='*70}")
        print(f"{'Layer':<35} {'Fault Impact':>15} {'Classification':>15}")
        print(f"{'-'*70}")

        for layer in prunable_layers:
            impact = self.fault_impacts.get(layer, 0.0)
            classification = self.get_layer_classification(layer)
            print(f"{layer:<35} {impact:>14.2f}% {classification:>15}")

        print(f"{'='*70}\n")
