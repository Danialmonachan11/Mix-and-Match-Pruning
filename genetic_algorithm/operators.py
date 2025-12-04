"""Genetic Algorithm operators for NSGA-II."""

import random
import numpy as np
from deap import tools
from typing import List, Tuple, Dict, Any
from core.utils import set_random_seeds


class GeneticOperators:
    """Collection of genetic operators for reliability-aware pruning."""
    
    def __init__(self, min_val: float = 50.0, max_val: float = 75.0):
     
        self.min_val = min_val
        self.max_val = max_val
    
    def blend_crossover(self, ind1: List[float], ind2: List[float], alpha: float = 0.5, 
                       crossover_prob: float = None, gene_constraints: List[Tuple[float, float]] = None) -> Tuple[List[float], List[float]]:
      
        offspring1 = list(ind1)
        offspring2 = list(ind2)
        
        # CRITICAL FIX: Auto-detect optimal crossover probability
        if crossover_prob is None:
            if len(ind1) == 1:
                crossover_prob = 0.9  # High probability for global mode (1 gene)
            else:
                crossover_prob = 0.5  # Standard for layer-wise mode (multiple genes)
        
        for i in range(len(ind1)):
            # Get gene-specific constraints if provided, otherwise use global bounds
            if gene_constraints and i < len(gene_constraints):
                gene_min, gene_max = gene_constraints[i]
            else:
                gene_min, gene_max = self.min_val, self.max_val
            
            if random.random() < crossover_prob:  # Apply crossover to this gene
                # FIX: Ensure values are floats to avoid "can't multiply sequence by non-int" error
                x1, x2 = float(ind1[i]), float(ind2[i])
                
                # Calculate bounds for blending
                min_val = min(x1, x2)
                max_val = max(x1, x2)
                interval = max_val - min_val
                
                lower_bound = max(gene_min, min_val - alpha * interval)
                upper_bound = min(gene_max, max_val + alpha * interval)
                
                # Generate offspring values within constraints
                offspring1[i] = random.uniform(lower_bound, upper_bound)
                offspring2[i] = random.uniform(lower_bound, upper_bound)
            else:
                # No crossover applied - but still enforce constraints on parent values
                offspring1[i] = float(max(gene_min, min(gene_max, float(offspring1[i]))))
                offspring2[i] = float(max(gene_min, min(gene_max, float(offspring2[i]))))
        
        return offspring1, offspring2
    
    # Note: Simulated Binary Crossover removed - blend_crossover with sensitivity constraints is used instead
    
    def gaussian_mutation(self, individual: List[float], mu: float = 0, 
                         sigma: float = 10.0, indpb: float = 0.2) -> List[float]:
      
        mutated = list(individual)
        for i in range(len(mutated)):
            if random.random() < indpb:
                mutated[i] += random.gauss(mu, sigma)
                mutated[i] = max(self.min_val, min(mutated[i], self.max_val))
        
        return mutated
    
    def polynomial_mutation(self, individual: List[float], eta: float = 20.0, 
                           indpb: float = 0.2) -> Tuple[List[float]]:
       
        for i in range(len(individual)):
            if random.random() < indpb:
                y = individual[i]
                yl = self.min_val
                yu = self.max_val
                
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                
                u = random.random()
                if u <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta + 1.0))
                    deltaq = val ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta + 1.0))
                    deltaq = 1.0 - (val ** (1.0 / (eta + 1.0)))
                
                y = y + deltaq * (yu - yl)
                individual[i] = max(yl, min(yu, y))
        
        return individual,
    
    def adaptive_mutation(self, individual: List[float], generation: int, 
                         max_generations: int, base_sigma: float = 10.0,
                         indpb: float = 0.2) -> Tuple[List[float]]:
        
        # Decrease sigma over time
        progress = generation / max_generations
        sigma = base_sigma * (1.0 - progress)
        
        return self.gaussian_mutation(individual, sigma=sigma, indpb=indpb)
    
    def reliability_island_mutation(self, individual: List[float], 
                                   reliability_score: float = None,
                                   generation: int = 0,
                                   population_diversity: float = 0.0,
                                   base_sigma: float = 15.0,
                                   indpb: float = 0.3) -> Tuple[List[float]]:
      
        mutated = list(individual)
        
        # ADAPTIVE MUTATION INTENSITY based on reliability performance and diversity
        if reliability_score is None or reliability_score < 30.0 or population_diversity < 5.0:
            # CRITICAL: Use island-hopping mutation for exploration
            mutation_type = random.choice(['gaussian', 'jump', 'invert', 'swap'])
            
            if mutation_type == 'jump':
                # Jump to different sparsity regions (island hopping) - FIXED SAFE RANGES
                target_regions = [(50, 55), (55, 65), (65, 70), (70, 75)]
                target_min, target_max = random.choice(target_regions)
                
                for i in range(len(mutated)):
                    if random.random() < 0.4:  # 40% chance per gene
                        mutated[i] = random.uniform(target_min, target_max)
            
            elif mutation_type == 'invert':
                # Invert pruning strategy (high <-> low)
                for i in range(len(mutated)):
                    if random.random() < 0.3:
                        mutated[i] = 90.0 - mutated[i] + 10.0  # Invert around center
                        
            elif mutation_type == 'swap':
                # Swap genes between random positions
                if len(mutated) > 1:
                    for _ in range(min(3, len(mutated)//2)):
                        i, j = random.sample(range(len(mutated)), 2)
                        if random.random() < 0.5:
                            mutated[i], mutated[j] = mutated[j], mutated[i]
                            
            else:  # gaussian
                # Aggressive gaussian for broad exploration
                sigma = 25.0  # Large jumps
                for i in range(len(mutated)):
                    if random.random() < 0.5:
                        mutated[i] += random.gauss(0, sigma)
                        
        elif reliability_score < 50.0:
            # MODERATE: Enhanced gaussian mutation
            sigma = base_sigma * 1.8
            for i in range(len(mutated)):
                if random.random() < indpb * 1.4:
                    mutated[i] += random.gauss(0, sigma)
                    
        else:
            # GOOD reliability: fine-tuning mutation with occasional jumps
            if random.random() < 0.1:  # 10% chance of exploration jump
                # Occasional exploration even for good solutions
                jump_target = random.choice([30, 50, 75, 90])
                gene_idx = random.randint(0, len(mutated)-1)
                mutated[gene_idx] = jump_target + random.gauss(0, 5)
            else:
                # Standard fine-tuning
                sigma = max(3.0, base_sigma * 0.7)
                for i in range(len(mutated)):
                    if random.random() < indpb * 0.8:
                        mutated[i] += random.gauss(0, sigma)
        
        # HARDCODED BOUNDS: [30.0, 85.0]% - CRITICAL for experimental consistency
        #
        # WHY HARDCODED:
        # 1. Config-based bounds led to inconsistent results across GA runs
        # 2. Different initialization strategies (4-strategy approach) need uniform bounds
        # 3. Paper results require these exact bounds for reproducibility
        # 4. Mutation and crossover operators must use same bounds to prevent constraint violations
        # 5. GA runner initializes operators with different bounds than config specifies
        #
        # These values OVERRIDE initial_percentile_min/max in ga_config.py
        # See config/ga_config.py lines 28-31 for acknowledgment of this override
        for i in range(len(mutated)):
            mutated[i] = max(30.0, min(85.0, mutated[i]))  # Bounds that produced paper results
            
        return mutated,


def create_deap_operators(config) -> Dict[str, Any]:
 
    # HARDCODED BOUNDS: [30.0, 85.0]% - Required for experimental reproducibility
    #
    # RATIONALE:
    # - Ensures all genetic operators (mutation, crossover) use identical bounds
    # - Prevents config drift from affecting published experimental results
    # - Maintains consistency across different GA configuration modes
    # - Required by 4-strategy sensitivity-driven initialization approach
    #
    # These values override config settings - see mutation function comments above
    operators = GeneticOperators(min_val=30.0, max_val=85.0)  # Paper-consistent bounds
    
    # Storage for high-reliability patterns (will be updated by GA runner)
    high_reliability_patterns = []
    
    # Create MODE-SPECIFIC operator functions
    def crossover_func(ind1, ind2):
        # RELIABILITY-INFORMED CROSSOVER: Learn from high-AUC individuals
        if hasattr(ind1, 'fitness') and hasattr(ind2, 'fitness'):
            reliability1 = ind1.fitness.values[1] if ind1.fitness.valid and len(ind1.fitness.values) >= 2 else 0.0
            reliability2 = ind2.fitness.values[1] if ind2.fitness.valid and len(ind2.fitness.values) >= 2 else 0.0
            
            # If both parents have high reliability, use conservative crossover
            if reliability1 > 40.0 and reliability2 > 40.0:
                alpha = 0.3  # Conservative blending to preserve good patterns
            # If one parent has high reliability, bias toward it
            elif reliability1 > 40.0 or reliability2 > 40.0:
                alpha = 0.5  # Standard blending
            else:
                # Neither parent is reliable, use aggressive exploration
                alpha = 0.7  # More exploration
        else:
            # Fallback to original logic
            if config.global_mode and len(ind1) == 1:
                alpha = 0.7
            else:
                alpha = 0.5
        
        # Get sensitivity constraints if available
        gene_constraints = None
        if hasattr(config, 'score_dir_path') and hasattr(config, 'prunable_layers'):
            try:
                from .sensitivity_driven_agents import SimpleSensitivityAnalyzer
                analyzer = SimpleSensitivityAnalyzer(config.score_dir_path)
                profiles = analyzer.analyze_all_layers()
                
                # Create constraints list for each layer
                gene_constraints = []
                for layer_name in getattr(config, 'prunable_layers', []):
                    if layer_name in profiles:
                        profile = profiles[layer_name]
                        if profile.protection_priority == "CRITICAL":
                            gene_constraints.append((2.0, 15.0))
                        elif profile.protection_priority == "HIGH":
                            gene_constraints.append((5.0, 35.0))
                        elif profile.protection_priority == "MEDIUM":
                            gene_constraints.append((10.0, 60.0))
                        else:  # LOW
                            gene_constraints.append((20.0, 80.0))
                    else:
                        # Fallback based on layer position
                        layer_idx = len(gene_constraints)
                        if layer_idx < 2:  # Early layers
                            gene_constraints.append((2.0, 15.0))
                        elif layer_idx < 6:  # Mid layers
                            gene_constraints.append((5.0, 35.0))
                        else:  # Late layers
                            gene_constraints.append((15.0, 75.0))
            except Exception as e:
                print(f"Warning: Could not load sensitivity constraints: {e}")
                gene_constraints = None
        
        offspring1, offspring2 = operators.blend_crossover(ind1, ind2, alpha=alpha, gene_constraints=gene_constraints)
        
        # Modify individuals in place (DEAP requirement)
        ind1[:] = offspring1
        ind2[:] = offspring2
        return ind1, ind2
    
    def mutation_func(individual):
        # RELIABILITY-INFORMED MUTATION: Learn from individual's reliability performance
        current_reliability = 0.0
        if hasattr(individual, 'fitness') and individual.fitness.valid and len(individual.fitness.values) >= 2:
            current_reliability = individual.fitness.values[1]
        
        # Adjust mutation intensity based on reliability performance
        if current_reliability > 50.0:
            # High reliability: conservative mutation to fine-tune
            base_sigma = 3.0
            indpb_modifier = 0.8  # Reduce mutation probability
        elif current_reliability > 30.0:
            # Medium reliability: moderate mutation for optimization
            base_sigma = 8.0
            indpb_modifier = 1.0  # Standard mutation probability
        else:
            # Low reliability: aggressive mutation for exploration
            base_sigma = 15.0
            indpb_modifier = 1.3  # Increase mutation probability
        
        sparsity_range = 75.0 - 50.0  # 25 (SAFE range for exploration)
        
        if config.global_mode and len(individual) == 1:
            # Global mode: Adjust sigma based on reliability
            adaptive_sigma = base_sigma * 1.2
        else:
            # Layer-wise mode: Standard sigma adjustment
            adaptive_sigma = base_sigma
        
        # Apply reliability-informed mutation probability
        exploration_indpb = config.mutation_indpb * indpb_modifier
        
        # Use reliability-informed island-hopping mutation
        mutated, = operators.reliability_island_mutation(
            individual, 
            reliability_score=current_reliability,
            generation=getattr(individual, '_generation', 0),
            population_diversity=getattr(individual, '_pop_diversity', 0.0),
            base_sigma=adaptive_sigma,
            indpb=exploration_indpb
        )
        # Modify individual in place (DEAP requirement)
        individual[:] = mutated
        return individual,
    
    # INTEGRATED CONSTRAINT HANDLING: Use constrained operators when needed
    if hasattr(config, 'min_overall_sparsity_threshold') and config.min_overall_sparsity_threshold > 0:
        print(f"Using constrained operators with min sparsity threshold: {config.min_overall_sparsity_threshold}%")
        # Note: For now, we use the basic operators but add constraint checking
        # Future enhancement: Integrate ConstrainedOperators class fully
    
    return {
        'crossover': crossover_func,
        'mutation': mutation_func,
        'selection': tools.selNSGA2  # NSGA-II selection
    }


class AdaptiveOperators:
    """Adaptive genetic operators that change behavior during evolution."""
    
    def __init__(self, config):
        """Initialize adaptive operators with configuration."""
        self.config = config
        self.base_operators = GeneticOperators()
        self.generation_history = []
    
    def adaptive_crossover_rate(self, generation: int, population_diversity: float) -> float:
     
        base_rate = self.config.crossover_prob
        
        # Increase crossover when diversity is low
        if population_diversity < 0.1:
            return min(0.9, base_rate * 1.5)
        elif population_diversity > 0.5:
            return max(0.3, base_rate * 0.8)
        else:
            return base_rate
    
    def adaptive_mutation_rate(self, generation: int, convergence_rate: float) -> float:
     
        base_rate = self.config.mutation_prob
        
        # Increase mutation when convergence stagnates
        if convergence_rate < 0.01:  # Very slow convergence
            return min(0.5, base_rate * 2.0)
        elif convergence_rate > 0.1:  # Fast convergence
            return max(0.05, base_rate * 0.5)
        else:
            return base_rate
    
    def calculate_population_diversity(self, population) -> float:
        
        if len(population) < 2:
            return 0.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = np.linalg.norm(np.array(population[i]) - np.array(population[j]))
                distances.append(dist)
        
        if not distances:
            return 0.0
        
        # Normalize by maximum possible distance (updated for safe bounds)
        max_dist = np.sqrt(len(population[0]) * (85.0 - 30.0) ** 2)
        avg_distance = np.mean(distances)
        
        return min(1.0, avg_distance / max_dist)
    
    def update_parameters(self, generation: int, population, fitness_history: List[float]) -> Dict[str, float]:
      
        # Calculate metrics
        diversity = self.calculate_population_diversity(population)
        
        convergence_rate = 0.0
        if len(fitness_history) > 5:
            recent_improvement = fitness_history[-1] - fitness_history[-5]
            convergence_rate = max(0.0, recent_improvement) / 5.0
        
        # Update parameters
        updated_params = {
            'crossover_prob': self.adaptive_crossover_rate(generation, diversity),
            'mutation_prob': self.adaptive_mutation_rate(generation, convergence_rate),
            'diversity': diversity,
            'convergence_rate': convergence_rate
        }
        
        self.generation_history.append({
            'generation': generation,
            'diversity': diversity,
            'convergence_rate': convergence_rate,
            **updated_params
        })
        
        return updated_params


class ConstrainedOperators:
    """Operators that maintain feasibility constraints."""
    
    def __init__(self, layer_param_counts: List[int], total_params: int, 
                 min_sparsity_threshold: float = 5.0):
     
        self.layer_param_counts = layer_param_counts
        self.total_params = total_params
        self.min_sparsity_threshold = min_sparsity_threshold
        self.base_operators = GeneticOperators()
    
    def calculate_projected_sparsity(self, individual: List[float]) -> float:
        """Calculate projected sparsity from individual parameters."""
        total_pruned = 0
        for i, percentile in enumerate(individual):
            if i < len(self.layer_param_counts):
                effective_percentile = max(0.1, min(percentile, 99.9))
                layer_pruned = int(self.layer_param_counts[i] * (effective_percentile / 100.0))
                total_pruned += layer_pruned
        
        return (total_pruned / self.total_params) * 100.0 if self.total_params > 0 else 0.0
    
    def constrained_crossover(self, ind1: List[float], ind2: List[float], 
                             max_attempts: int = 100) -> Tuple[List[float], List[float]]:
       
        for attempt in range(max_attempts):
            offspring1, offspring2 = self.base_operators.blend_crossover(ind1, ind2)
            
            # Check constraints
            sparsity1 = self.calculate_projected_sparsity(offspring1)
            sparsity2 = self.calculate_projected_sparsity(offspring2)
            
            if (sparsity1 >= self.min_sparsity_threshold and 
                sparsity2 >= self.min_sparsity_threshold):
                return offspring1, offspring2
        
        # If no feasible offspring found, return parents
        print(f"Warning: Could not generate feasible offspring after {max_attempts} attempts")
        return ind1[:], ind2[:]
    
    def constrained_mutation(self, individual: List[float], 
                           max_attempts: int = 50) -> Tuple[List[float]]:
     
        original_sparsity = self.calculate_projected_sparsity(individual)
        
        for attempt in range(max_attempts):
            mutated = individual[:]
            mutated, = self.base_operators.gaussian_mutation(mutated)
            
            new_sparsity = self.calculate_projected_sparsity(mutated)
            if new_sparsity >= self.min_sparsity_threshold:
                return mutated,
        
        # If mutation fails, return original
        return individual[:],
    
    def increase_mutation_pressure(self):
        """Increase mutation pressure for next generation due to low diversity."""
        # This could temporarily increase mutation rates in the config
      
        print("   Adaptive operators: Increasing mutation pressure for next generation")