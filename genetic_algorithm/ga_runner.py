"""Main genetic algorithm runner with NSGA-II implementation."""

import os
import random
import timeit
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from deap import base, creator, tools, algorithms

from config.ga_config import GAConfig
from core.models import create_model_from_config
from core.data import get_data_loaders
from core.utils import set_random_seeds, cleanup_memory, get_layer_score_files_map
from benchmarking.reliability.reliability_test import ReliabilityTester
from visualization.ga_plots import create_ga_plotter
from .initialization import PopulationInitializer, SensitivityInformedPopulationInitializer, create_deap_individuals
from .operators import create_deap_operators, AdaptiveOperators
from .evaluation import create_nsga2_fitness_evaluator
from .agents import get_sensitivity_based_layer_ordering  # CRITICAL FIX: Use sensitivity-based ordering
# RL Enhancement imports (optional)
try:
    from .rl_adaptive_operators import RLAdaptiveGeneticOperators
    from .rl_parameter_controller import RLParameterController
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("RL components not available - using standard GA")


class MetaLearningController:
    """Autonomous meta-learning system for GA self-improvement."""
    
    def __init__(self, ga_config):
        self.ga_config = ga_config
        self.adaptation_history = []
        self.success_threshold = 50.0  # 50% reliability considered success
        self.exceptional_threshold = 70.0  # 70% reliability considered exceptional
        
    def analyze_progress(self, generation_stats: List[Dict], generation: int) -> Dict:
        """Analyze current progress and recommend adaptations."""
        if len(generation_stats) < 5:
            return {'action': 'continue', 'reason': 'insufficient_data'}
        
        recent_stats = generation_stats[-5:]
        reliabilities = [stat.get('reliability_mean', 0) for stat in recent_stats]
        current_rel = reliabilities[-1]
        rel_trend = reliabilities[-1] - reliabilities[0]  # 5-generation trend
        
        # SUCCESS DETECTION
        if current_rel >= self.exceptional_threshold:
            return {
                'action': 'early_success',
                'reason': f'exceptional_reliability_{current_rel:.1f}%',
                'recommendation': 'extend_generations_for_refinement'
            }
        elif current_rel >= self.success_threshold and rel_trend > 0:
            return {
                'action': 'success_detected', 
                'reason': f'good_reliability_{current_rel:.1f}%_trending_up',
                'recommendation': 'continue_with_intensified_local_search'
            }
        
        # STAGNATION DETECTION (enhanced)
        if abs(rel_trend) < 1.0 and current_rel < 40.0 and generation > 15:
            return {
                'action': 'stagnation_critical',
                'reason': f'no_progress_{rel_trend:.1f}%_over_5_gens',
                'recommendation': 'aggressive_restart_needed'
            }
        
        # SLOW PROGRESS DETECTION
        if rel_trend < 2.0 and current_rel < 30.0 and generation > 20:
            return {
                'action': 'slow_progress',
                'reason': f'insufficient_progress_{rel_trend:.1f}%',
                'recommendation': 'increase_exploration_pressure'
            }
        
        return {'action': 'continue', 'reason': 'normal_progress'}
    
    def get_adaptive_parameters(self, analysis_result: Dict, generation: int) -> Dict:
        """Get adapted GA parameters based on analysis."""
        adaptations = {}
        
        if analysis_result['action'] == 'success_detected':
            # Success detected: shift to local refinement
            adaptations = {
                'mutation_prob': max(0.2, self.ga_config.mutation_prob * 0.7),
                'mutation_sigma': max(5.0, self.ga_config.mutation_sigma * 0.6),
                'crossover_prob': min(0.9, self.ga_config.crossover_prob * 1.2),
                'reason': 'local_refinement_mode'
            }
        
        elif analysis_result['action'] == 'stagnation_critical':
            # Critical stagnation: maximum exploration
            adaptations = {
                'mutation_prob': min(0.9, self.ga_config.mutation_prob * 1.5),
                'mutation_sigma': min(40.0, self.ga_config.mutation_sigma * 2.0), 
                'crossover_prob': max(0.4, self.ga_config.crossover_prob * 0.8),
                'reason': 'maximum_exploration_mode'
            }
        
        elif analysis_result['action'] == 'slow_progress':
            # Slow progress: moderate exploration boost
            adaptations = {
                'mutation_prob': min(0.8, self.ga_config.mutation_prob * 1.3),
                'mutation_sigma': min(30.0, self.ga_config.mutation_sigma * 1.5),
                'reason': 'exploration_boost_mode'
            }
        
        return adaptations
    
    def should_extend_evolution(self, generation_stats: List[Dict], max_generations: int, current_gen: int) -> bool:
        """Decide if evolution should be extended beyond max generations."""
        if current_gen < max_generations:
            return False
            
        if len(generation_stats) < 5:
            return False
            
        # Check if we're making good progress near the end
        recent_progress = generation_stats[-5:]
        reliabilities = [stat.get('reliability_mean', 0) for stat in recent_progress]
        
        # Extend if we're trending upward and haven't reached exceptional performance
        trend = reliabilities[-1] - reliabilities[0]
        current_rel = reliabilities[-1]
        
        if trend > 2.0 and current_rel < 75.0:  # Still improving and room for growth
            return True
        
        return False


class ReliabilityAwareGA:
    """Main genetic algorithm runner for reliability-aware pruning."""
    
    def __init__(self, ga_config: GAConfig, model_config):
       
        self.ga_config = ga_config
        self.model_config = model_config
        
        # Initialize components
        self.base_model = None
        self.train_loader = None
        self.val_loader = None
        self.reliability_tester = None
        self.fitness_evaluator = None
        
        # GA components
        self.toolbox = None
        self.population = None
        self.hall_of_fame = None
        self.logbook = None
        
        # Results tracking
        self.all_evaluation_results = []
        self.generation_stats = []
        
        # Enhanced tracking for reliability trends
        self.reliability_trend = []  # Track actual reliability over generations
        self.accuracy_trend = []     # Track actual accuracy over generations  
        self.fitness_trend = []      # Track fitness scores over generations
        
        # Initialize plotter for real-time visualization
        self.plotter = create_ga_plotter(save_plots=True, show_plots=False)
        
        # Track restart history to avoid repeated strategies
        self.restart_count = 0
        self.explored_regions = []  # Track which sparsity regions we've explored
        
        # Simple tracking for basic diversity management
        self.diversity_injection_count = 0
        
        # COMPLETELY DISABLE RL for layerwise mode to ensure pure sensitivity approach
        # RL can interfere with sensitivity-informed decisions and tournament selection
        print("RL Controller DISABLED for layerwise mode - Using pure sensitivity-aware approach")
        print("All diversity management will use sensitivity-informed strategies only")
        self.rl_controller = None
        
        # Initialize everything
        self._initialize_components()
        self._setup_deap()
    
    def _initialize_components(self):
        """Initialize all necessary components."""
        print("=== Initializing Reliability-Aware GA Components ===")
        
        # Set random seeds
        set_random_seeds(42)
        
        # Create model
        print("Loading base model...")
        self.base_model = create_model_from_config(self.model_config)
        
        # Create data loaders
        print("Setting up data loaders...")
        self.train_loader, self.val_loader = get_data_loaders(self.model_config)
        
        # Initialize reliability tester
        print("Initializing reliability tester...")
        self.reliability_tester = ReliabilityTester(
            enable_parallel=True,
            max_workers=4
        )
        
        # Create fitness evaluator
        print("Creating fitness evaluator...")
        self.fitness_evaluator = create_nsga2_fitness_evaluator(
            base_model=self.base_model,
            config=self.ga_config,
            train_dataloader=self.train_loader,
            val_dataloader=self.val_loader,
            reliability_tester=self.reliability_tester,
            global_mode=self.ga_config.global_mode
        )
        
        print("All components initialized successfully")
    
    def _setup_deap(self):
        """Setup DEAP genetic algorithm framework."""
        print("Setting up DEAP framework...")
        
        # Create fitness and individual classes
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        creator.create("FitnessMulti", base.Fitness, weights=self.ga_config.fitness_weights)
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        # Initialize toolbox
        self.toolbox = base.Toolbox()
        
        # Get prunable layers and setup initialization
        score_files_map = get_layer_score_files_map(
            self.ga_config.score_dir_path,
            self.base_model.state_dict()
        )

        # CRITICAL FIX: Use sensitivity-based ordering instead of alphabetical
        # This ensures GA and benchmark use the SAME layer ordering
        available_layers = [
            name for name in self.base_model.state_dict().keys()
            if ('features' in name or 'classifier' in name) and 'weight' in name
            and name in score_files_map
        ]
        prunable_layers = get_sensitivity_based_layer_ordering(available_layers, self.ga_config.score_dir_path)
        print(f"\nUsing sensitivity-based layer ordering ({len(prunable_layers)} layers)")
        
        layer_param_counts = [
            self.base_model.state_dict()[name].numel() 
            for name in prunable_layers
        ]
        
        total_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        
        # Create sensitivity-informed population initializer
        print("Using sensitivity-informed population initialization from Generation 0")
        # Check for fault-aware mode
        fault_impact_path = getattr(self.ga_config, 'fault_impact_filepath', None)
        if fault_impact_path:
            print(f"FAULT-AWARE MODE: Using fault-impacts from {fault_impact_path}")

        initializer = SensitivityInformedPopulationInitializer(
            prunable_layer_names=prunable_layers,
            layer_param_counts=layer_param_counts,
            total_params=total_params,
            score_dir_path=self.ga_config.score_dir_path,
            global_mode=self.ga_config.global_mode,
            config=self.ga_config,
            fault_impact_filepath=fault_impact_path  # Pass fault-impact filepath
        )
        
        # Store population initializer for diversity injection
        self.population_initializer = initializer
        
        # Add prunable layers to config for sensitivity-aware crossover
        self.ga_config.prunable_layers = prunable_layers
        
        # Register genetic operators
        individual_creator = create_deap_individuals(initializer, self.ga_config)
        self.deap_operators = create_deap_operators(self.ga_config)  # DEAP-wrapped functions
        
        # Create raw operators object for RL access - FIXED SAFE RANGES
        from .operators import GeneticOperators
        max_allowed = getattr(self.ga_config, 'max_overall_sparsity_threshold', 75.0)
        min_allowed = getattr(self.ga_config, 'min_overall_sparsity_threshold', 50.0)
        self.raw_operators = GeneticOperators(min_val=min_allowed, max_val=max_allowed)
        
        # Initialize RL-enhanced operators if available (DISABLED for layerwise mode)
        # Force disable RL genetic operators for layerwise to use reliability-informed operators
        self.rl_genetic_operators = None
        print("RL-Enhanced genetic operators DISABLED for layerwise mode - Using reliability-informed operators")
        
        self.toolbox.register("individual", individual_creator)
        # FIX: Ensure x is always a proper list to avoid "can't multiply sequence by non-int" error
        self.toolbox.register("individual_from_list", lambda x: creator.Individual([float(v) for v in x]))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.fitness_evaluator.evaluate_individual)
        self.toolbox.register("mate", self.deap_operators['crossover'])
        
        # Use RL-enhanced mutation if available, otherwise standard mutation
        if hasattr(self, 'rl_genetic_operators') and self.rl_genetic_operators:
            self.toolbox.register("mutate", self._rl_enhanced_mutation)
        else:
            self.toolbox.register("mutate", self.deap_operators['mutation'])
            
        self.toolbox.register("select", self.deap_operators['selection'])
        
        # Initialize ENHANCED hall of fame and statistics
        # Enhanced Pareto front with diversity preservation for 10-20 high-quality solutions
        # Bias toward reliability-focused solutions
        self.hall_of_fame = tools.ParetoFront(similar=self._is_too_similar)
        self.reliability_hall_of_fame = tools.HallOfFame(maxsize=10, similar=self._is_too_similar)  # Top 10 by reliability
        self.logbook = tools.Logbook()
        
        print("DEAP framework setup complete")
    
    def _rl_enhanced_mutation(self, individual):
        """RL-enhanced mutation that adapts based on population state."""
        if hasattr(self, 'rl_genetic_operators') and self.rl_genetic_operators:
            # Get current population statistics for RL guidance
            if hasattr(self, 'generation_stats') and self.generation_stats:
                current_stats = self.generation_stats[-1]
            else:
                # Fallback stats if no generation stats available yet
                current_stats = {
                    'diversity': 5.0,
                    'reliability_mean': 20.0,
                    'generation': 0
                }
            
            # Use RL-guided adaptive mutation
            mutated_tuple = self.rl_genetic_operators.adaptive_mutation(individual, current_stats)
            
            # Update individual in-place (DEAP requirement)
            if mutated_tuple and len(mutated_tuple) > 0:
                individual[:] = mutated_tuple[0]  # Update individual in-place
            
            # Update RL performance tracking if we have previous stats
            if len(self.generation_stats) > 1:
                self.rl_genetic_operators.update_performance(current_stats)
            
            return individual,  # Return tuple as required by DEAP
        else:
            # Fallback to standard mutation if RL not available
            return self.deap_operators['mutation'](individual)
    
    def run_evolution(self) -> Tuple[List, tools.Logbook, tools.ParetoFront, Dict]:
       
        print(f"\n=== Starting NSGA-II Evolution ===")
        print(f"Population size: {self.ga_config.population_size}")
        print(f"Generations: {self.ga_config.num_generations}")
        print(f"Objectives: {len(self.ga_config.fitness_weights)}")
        
        start_time = timeit.default_timer()
        
        # Check for checkpoint - DISABLED to avoid pickle multiprocessing errors
        start_gen = 0
        # if self._load_checkpoint():
        #     start_gen = len(self.logbook)
        #     print(f"Resumed from checkpoint at generation {start_gen}")
        # else:
        # Create initial population using sensitivity-informed strategies
        # Use fault-aware initialization if available
        if self.population_initializer.use_fault_aware_init:
            print("Creating FAULT-AWARE initial population...")
            diverse_individuals = self.population_initializer.create_fault_aware_population(
                self.ga_config.population_size,
                fault_aware_ratio=0.7  # 70% fault-aware, 30% random
            )
        else:
            print("Creating sensitivity-informed initial population...")
            diverse_individuals = self.population_initializer.create_sensitivity_informed_population(
                self.ga_config.population_size
            )
        # FIX: Ensure all values are floats to avoid "can't multiply sequence by non-int" error
        self.population = [creator.Individual([float(v) for v in ind]) for ind in diverse_individuals]
        self._evaluate_population(self.population, generation=0)
        
        # Evolution loop with autonomous extension capability
        original_generations = self.ga_config.num_generations
        extended_generations = 0
        
        try:
            gen = start_gen
            while gen < self.ga_config.num_generations:
                current_total = original_generations + extended_generations
                print(f"\n--- Generation {gen + 1}/{current_total} ---")
                
                # Generate offspring
                offspring = self._generate_offspring()
                print(f"DEBUG: Generated {len(offspring)} offspring")
                
                # Debug: Show offspring fitness status
                invalid_count = sum(1 for ind in offspring if not ind.fitness.valid)
                print(f"DEBUG: {invalid_count} offspring have invalid fitness (should be {len(offspring)})")
                
                # Evaluate offspring
                self._evaluate_population(offspring, generation=gen + 1)
                
                # Select next generation using reliability-biased selection
                combined_population = self.population + offspring
                
                # Filter out individuals with invalid fitness
                valid_population = [
                    ind for ind in combined_population 
                    if ind.fitness.valid and len(ind.fitness.values) == 2
                ]
                
                # If we don't have enough valid individuals, fill with random ones
                while len(valid_population) < self.ga_config.population_size:
                    new_ind = self.toolbox.individual()
                    # Give it a default poor fitness so selection can handle it
                    new_ind.fitness.values = (0.0, 0.0)
                    valid_population.append(new_ind)
                
                # Use reliability-biased selection instead of pure NSGA-II
                self.population = self._reliability_biased_selection(
                    valid_population, 
                    self.ga_config.population_size
                )
                
                # Update hall of fame and reliability-focused hall of fame with elite preservation
                self.hall_of_fame.update(self.population)
                
                # Update reliability-focused hall of fame (sort by reliability fitness only)
                reliability_sorted_pop = sorted([ind for ind in self.population if ind.fitness.valid], 
                                               key=lambda ind: ind.fitness.values[1], reverse=True)
                self.reliability_hall_of_fame.update(reliability_sorted_pop)
                
                # ELITE PRESERVATION: Ensure high-reliability elites survive to next generation
                self._preserve_high_reliability_elites(gen + 1)
                
                # Record statistics
                self._record_generation_stats(gen + 1)
                
                # Save checkpoint - DISABLED to avoid pickle multiprocessing errors
                # if (gen + 1) % self.ga_config.checkpoint_frequency == 0:
                #     self._save_checkpoint(gen + 1)
                
                # Print progress
                self._print_generation_summary(gen + 1)
                
                # META-LEARNING SYSTEM: Autonomous adaptation every 5 generations
                # Disabled - meta_controller functionality removed
                # if (gen + 1) % 5 == 0 and gen > 5:
                #     self._apply_meta_learning_adaptations(gen + 1)
                
                # Check for stagnation and restart if needed
                if (gen + 1) % 5 == 0 and gen > 10:  # Start checking after gen 10
                    restart_triggered = self._check_stagnation_and_restart(gen + 1)
                    if restart_triggered:
                        print("Population restarted - continuing evolution with fresh diversity")
                
                # AUTO-EXTENSION: Check if we should extend evolution at the planned end
                # Disabled - meta_controller functionality removed
                # if gen + 1 == original_generations and not extended_generations:
                #     if self.meta_controller.should_extend_evolution(self.generation_stats, original_generations, gen + 1):
                #         extend_by = min(20, int(original_generations * 0.4))  # Extend by up to 40% more
                #         extended_generations = extend_by
                #         self.ga_config.num_generations += extend_by
                #         print(f"AUTO-EXTENSION: Extending evolution by {extend_by} generations due to promising progress")
                #         print(f"New total generations: {self.ga_config.num_generations}")
                
                gen += 1
        
        except KeyboardInterrupt:
            print("\nEvolution interrupted by user")
        except Exception as e:
            import traceback
            print(f"\nEvolution stopped due to error: {e}")
            print("\n=== FULL TRACEBACK ===")
            traceback.print_exc()
            print("=" * 80)
        
        total_time = timeit.default_timer() - start_time
        print(f"\n=== Evolution Complete ===")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Pareto front size: {len(self.hall_of_fame)}")
        
        # Save final results (includes Pareto CSV creation)
        print("Saving final results...")
        self._save_final_results()
        
        # Generate comprehensive analysis plots
        print("\n=== Generating Comprehensive Analysis ===")
        self.plotter.plot_evolution_trends(self.generation_stats, "GENIE Reliability Monster")
        self.plotter.plot_fitness_vs_actual_reliability(self.all_evaluation_results, "GENIE Scoring Analysis")
        
        # Generate reliability monster report
        report = self.plotter.create_reliability_monster_report(
            self.generation_stats, self.all_evaluation_results, list(self.hall_of_fame)
        )
        print(report)
        
        # Create metrics map
        metrics_map = self._create_metrics_map()
        
        return self.population, self.logbook, self.hall_of_fame, metrics_map
    
    def _generate_offspring(self) -> List:
        """Generate offspring using tournament selection for parent selection."""
        offspring = []
        
        # Generate exactly population_size offspring using tournament selection
        while len(offspring) < self.ga_config.population_size:
            # Tournament selection for both parents (k=3 for balanced selection pressure)
            parent1 = tools.selTournament(self.population, 1, tournsize=3)[0]
            parent2 = tools.selTournament(self.population, 1, tournsize=3)[0]
            
            if random.random() < self.ga_config.crossover_prob:
                # Create copies and apply reliability-informed crossover
                child1 = self.toolbox.individual_from_list(list(parent1))
                child2 = self.toolbox.individual_from_list(list(parent2))
                
                # Apply crossover using registered crossover function
                self.toolbox.mate(child1, child2)
                
                # Mark fitness as invalid
                child1.fitness.delValues()
                child2.fitness.delValues()
                
                offspring.extend([child1, child2])
            else:
                # No crossover - create copies of tournament-selected parents
                child1 = self.toolbox.individual_from_list(list(parent1))
                child2 = self.toolbox.individual_from_list(list(parent2))
                
                # Mark fitness as invalid (they're still new individuals)
                child1.fitness.delValues()
                child2.fitness.delValues()
                
                offspring.extend([child1, child2])
        
        # Validate sensitivity constraints after crossover (debugging)
        if hasattr(self.ga_config, 'prunable_layers'):
            constraint_violations = 0
            for child in offspring:
                for i, value in enumerate(child):
                    if i < len(self.ga_config.prunable_layers):
                        layer_name = self.ga_config.prunable_layers[i]
                        # Check early layer constraints (most critical)
                        if ('features.0' in layer_name or 'features.4' in layer_name) and value > 20.0:
                            constraint_violations += 1
                            break
            
            if constraint_violations > 0:
                print(f"   WARNING: {constraint_violations}/{len(offspring)} offspring may violate sensitivity constraints")
            else:
                print(f"   SUCCESS: All {len(offspring)} offspring respect sensitivity constraints")
        
        # Apply mutation to all offspring
        for child in offspring:
            if random.random() < self.ga_config.mutation_prob:
                self.toolbox.mutate(child)
                # Fitness already invalidated above
        
        # Ensure exactly population_size offspring
        return offspring[:self.ga_config.population_size]

    def _reliability_biased_selection(self, population: List, k: int) -> List:
        """Reliability-biased selection that prioritizes high-reliability individuals."""
        if len(population) <= k:
            return population
        
        # Split population into reliability tiers
        high_reliability = []  # Top 30% by reliability
        medium_reliability = []  # Middle 40% by reliability  
        low_reliability = []  # Bottom 30% by reliability
        
        # Sort by reliability fitness (index 1)
        population_sorted = sorted(population, key=lambda ind: ind.fitness.values[1], reverse=True)
        
        n = len(population_sorted)
        high_cutoff = int(n * 0.3)
        medium_cutoff = int(n * 0.7)
        
        high_reliability = population_sorted[:high_cutoff]
        medium_reliability = population_sorted[high_cutoff:medium_cutoff]
        low_reliability = population_sorted[medium_cutoff:]
        
        # Reliability-biased selection proportions
        high_selected_count = min(len(high_reliability), int(k * 0.5))    # 50% from high reliability
        medium_selected_count = min(len(medium_reliability), int(k * 0.35)) # 35% from medium reliability
        low_selected_count = k - high_selected_count - medium_selected_count # Remaining from low reliability
        
        selected = []
        
        # Select from high reliability tier using NSGA-II for diversity
        if high_reliability and high_selected_count > 0:
            if len(high_reliability) <= high_selected_count:
                selected.extend(high_reliability)
            else:
                # Use NSGA-II within high reliability group
                selected.extend(tools.selNSGA2(high_reliability, high_selected_count))
        
        # Select from medium reliability tier using NSGA-II
        if medium_reliability and medium_selected_count > 0:
            if len(medium_reliability) <= medium_selected_count:
                selected.extend(medium_reliability)
            else:
                selected.extend(tools.selNSGA2(medium_reliability, medium_selected_count))
        
        # Select from low reliability tier using NSGA-II for diversity
        if low_reliability and low_selected_count > 0:
            if len(low_reliability) <= low_selected_count:
                selected.extend(low_reliability)
            else:
                selected.extend(tools.selNSGA2(low_reliability, low_selected_count))
        
        # If we still need more individuals, fill from the best available
        while len(selected) < k and len(selected) < len(population):
            remaining = [ind for ind in population if ind not in selected]
            if remaining:
                # Pick the best remaining individual by reliability
                best_remaining = max(remaining, key=lambda ind: ind.fitness.values[1])
                selected.append(best_remaining)
            else:
                break
        
        print(f"   Reliability-biased selection: {high_selected_count} high, {medium_selected_count} medium, {low_selected_count} low reliability individuals")
        return selected[:k]

    def _preserve_high_reliability_elites(self, generation: int) -> None:
        """Preserve high-reliability elite individuals across generations."""
        if not self.reliability_hall_of_fame or len(self.reliability_hall_of_fame) == 0:
            return
        
        # Get the best reliability elites from hall of fame
        elite_threshold = 50.0  # Minimum reliability threshold for elite status
        elite_candidates = [
            ind for ind in self.reliability_hall_of_fame
            if ind.fitness.valid and len(ind.fitness.values) >= 2 and ind.fitness.values[1] >= elite_threshold
        ]
        
        if not elite_candidates:
            return
        
        # Calculate how many elites to preserve (5-20% of population)
        elite_count = min(len(elite_candidates), max(1, int(self.ga_config.population_size * 0.1)))
        
        # Find worst individuals in current population to replace
        population_sorted = sorted(self.population, key=lambda ind: ind.fitness.values[1] if ind.fitness.valid else 0.0)
        worst_individuals = population_sorted[:elite_count]
        
        # Only replace if the elite is significantly better
        replacements_made = 0
        for i, worst_ind in enumerate(worst_individuals):
            if i < len(elite_candidates):
                elite_ind = elite_candidates[i]
                
                # Check if elite is significantly better than worst individual
                worst_reliability = worst_ind.fitness.values[1] if worst_ind.fitness.valid else 0.0
                elite_reliability = elite_ind.fitness.values[1]
                
                if elite_reliability > worst_reliability + 10.0:  # At least 10% better
                    # Replace worst with elite (create a copy)
                    elite_copy = self.toolbox.individual_from_list(list(elite_ind))
                    elite_copy.fitness.values = elite_ind.fitness.values
                    
                    # Find and replace the worst individual in population
                    worst_idx = self.population.index(worst_ind)
                    self.population[worst_idx] = elite_copy
                    replacements_made += 1
        
        if replacements_made > 0:
            print(f"   Elite preservation: {replacements_made} high-reliability elites (≥{elite_threshold}%) preserved in population")
    
    def _evaluate_population(self, population: List, generation: int):
        """Evaluate fitness for all individuals in population."""
        invalid_individuals = [ind for ind in population if not ind.fitness.valid]
        
        if invalid_individuals:
            print(f"Evaluating {len(invalid_individuals)} individuals...")
            
            fitnesses = []
            for i, individual in enumerate(invalid_individuals):
                print(f"Individual {i+1}/{len(invalid_individuals)}")
                print(f"DEBUG: Evaluating strategy: {individual[:3]}... (first 3 values)")
                
                try:
                    fitness = self.toolbox.evaluate(individual, generation)
                    print(f"DEBUG: Fitness result: {fitness}")
                    fitnesses.append(fitness)
                except Exception as e:
                    print(f"ERROR: Fitness evaluation failed: {e}")
                    print(f"ERROR: Individual causing issue: {individual}")
                    # Assign default poor fitness to continue
                    fitness = (0.0, 0.0)
                    fitnesses.append(fitness)
                
                # Store results for analysis (including actual metrics like old code)
                # Use same key format as evaluation.py to match cache
                if self.ga_config.global_mode:
                    strategy_key = f"global_{individual[0]:.3f}"
                else:
                    strategy_key = tuple(round(x, 2) for x in individual)  # Match evaluation.py rounding
                actual_metrics = self.fitness_evaluator.last_evaluation_metrics.get(strategy_key, {})
                print(f"DEBUG: Cache lookup for {strategy_key}: {len(actual_metrics)} metrics found")
                if actual_metrics:
                    print(f"DEBUG: Found metrics - acc: {actual_metrics.get('accuracy', 'N/A')}, rel: {actual_metrics.get('reliability', 'N/A')}")
                
                self.all_evaluation_results.append({
                    'generation': generation,
                    'individual_id': i,
                    'strategy': individual[:],
                    'fitness': fitness,
                    'accuracy': actual_metrics.get('accuracy', 0),
                    'sparsity': actual_metrics.get('sparsity', 0),
                    'latency': actual_metrics.get('latency', 0),
                    'reliability': actual_metrics.get('reliability', 0),
                    'applied_layer_percentages': actual_metrics.get('applied_layer_percentages', None)
                })
            
            # Assign fitness values
            for ind, fit in zip(invalid_individuals, fitnesses):
                ind.fitness.values = fit
    
    def _record_generation_stats(self, generation: int):
        """Record statistics for current generation."""
        if not self.population:
            return
        
        # Calculate multi-objective statistics (UPDATED: 2 objectives)
        stats = {}
        objective_names = ['accuracy', 'reliability']  # Removed sparsity and latency
        
        for i, obj_name in enumerate(objective_names):
            values = [ind.fitness.values[i] for ind in self.population if ind.fitness.valid and len(ind.fitness.values) > i]
            if values:
                stats[f'{obj_name}_mean'] = np.mean(values)
                stats[f'{obj_name}_std'] = np.std(values)
                stats[f'{obj_name}_min'] = np.min(values)
                stats[f'{obj_name}_max'] = np.max(values)
        
        # ALSO extract RAW ACCURACY for RL controller (not fitness values)
        if hasattr(self, 'fitness_evaluator') and hasattr(self.fitness_evaluator, 'last_evaluation_metrics'):
            raw_accuracies = []
            raw_reliabilities = []
            for individual in self.population:
                if hasattr(individual, 'fitness') and individual.fitness.valid:
                    # Get cache key
                    if self.ga_config.global_mode:
                        strategy_key = f"global_{individual[0]:.3f}"
                    else:
                        strategy_key = tuple(round(x, 2) for x in individual)
                    
                    if strategy_key in self.fitness_evaluator.last_evaluation_metrics:
                        metrics = self.fitness_evaluator.last_evaluation_metrics[strategy_key]
                        if 'accuracy' in metrics:
                            raw_accuracies.append(metrics['accuracy'])
                        if 'reliability' in metrics:
                            raw_reliabilities.append(metrics['reliability'])
            
            # Store raw values for RL controller
            if raw_accuracies:
                stats['raw_accuracy_mean'] = np.mean(raw_accuracies)
                stats['raw_accuracy_std'] = np.std(raw_accuracies)
            if raw_reliabilities:
                stats['raw_reliability_mean'] = np.mean(raw_reliabilities)
                stats['raw_reliability_std'] = np.std(raw_reliabilities)
        
        # DIVERSITY MONITORING: Calculate population diversity
        diversity = self._calculate_population_diversity()
        stats['population_diversity'] = diversity
        
        # PURE SENSITIVITY-AWARE diversity management (RL disabled for layerwise mode)
        # Trigger diversity injection based on simple criteria: low diversity OR periodic
        if diversity < 8.0 or generation % 3 == 0:
            print(f"Sensitivity-aware diversity injection triggered: div={diversity:.2f}, gen={generation}")
            self._inject_sensitivity_aware_diversity()
        
        # RELIABILITY ANALYSIS: Extract multi-level reliability for analysis (RL disabled)
        multi_level_stats = self._extract_multi_level_reliability_stats()
        stats.update(multi_level_stats)
        
        # Detect cliff collapses for reliability monitoring (no RL)
        cliff_detected = self._detect_cliff_collapse(multi_level_stats)
        if cliff_detected:
            print(f"   Reliability Alert: Cliff collapse detected! {cliff_detected}")
            # Store for analysis but no RL action needed
        
        stats['generation'] = generation
        stats['population_size'] = len(self.population)
        stats['hall_of_fame_size'] = len(self.hall_of_fame)

        self.generation_stats.append(stats)

        # Update logbook
        record = {
            'gen': generation,
            'nevals': len([ind for ind in self.population if ind.fitness.valid]),
            **stats
        }
        self.logbook.record(**record)

        # UPDATE BEST PATTERN for reliability-preserving operators
        if 'update_best_pattern' in self.deap_operators:
            # Find best reliable individual in current population
            best_reliable_ind = self._get_best_reliable_individual()
            if best_reliable_ind is not None:
                self.deap_operators['update_best_pattern'](list(best_reliable_ind))
    
    def _get_best_reliable_individual(self):
        """Get the individual with highest reliability from current population."""
        if not self.population:
            return None

        # Find individuals with valid fitness
        valid_individuals = [
            ind for ind in self.population
            if hasattr(ind, 'fitness') and ind.fitness.valid and len(ind.fitness.values) >= 2
        ]

        if not valid_individuals:
            return None

        # Return individual with highest reliability (fitness.values[1])
        best = max(valid_individuals, key=lambda ind: ind.fitness.values[1])
        return best

    def _calculate_population_diversity(self) -> float:

        if len(self.population) < 2:
            return 0.0
        
        total_distance = 0.0
        count = 0
        
        # For global mode (single parameter), scale diversity differently
        if self.ga_config.global_mode:
            # Extract just the global sparsity values
            values = [individual[0] for individual in self.population if len(individual) > 0]
            if len(values) < 2:
                return 0.0
            
            # Calculate standard deviation of global sparsity values
            mean_val = np.mean(values)
            std_val = np.std(values)
            diversity = std_val * 2.0  # Scale for interpretability
            
            print(f"    Global diversity: std={std_val:.2f}, scaled={diversity:.2f}, range={min(values):.1f}-{max(values):.1f}%")
            return diversity
        else:
            # Layer-wise mode: use Euclidean distance
            for i in range(len(self.population)):
                for j in range(i + 1, len(self.population)):
                    # Calculate Euclidean distance between individuals
                    distance = sum((a - b) ** 2 for a, b in zip(self.population[i], self.population[j])) ** 0.5
                    total_distance += distance
                    count += 1
            
            return total_distance / count if count > 0 else 0.0
    
    def _extract_multi_level_reliability_stats(self) -> dict:
        """
        Extract multi-level reliability statistics for degradation analysis.
        
        Returns:
            Dictionary with reliability stats at different fault levels
        """
        multi_level_stats = {}
        
        # Extract reliability at each fault level from population
        fault_levels = [10, 25, 50, 100, 150]
        
        for fault_level in fault_levels:
            reliabilities = []
            for individual in self.population:
                if hasattr(individual, 'fitness') and individual.fitness.valid:
                    # Try to extract per-level reliability from fitness evaluation
                    fitness_evaluator = getattr(self, 'fitness_evaluator', None)
                    if fitness_evaluator and hasattr(fitness_evaluator, 'last_evaluation_metrics'):
                        strategy_tuple = tuple(individual)
                        if strategy_tuple in fitness_evaluator.last_evaluation_metrics:
                            rel_per_level = fitness_evaluator.last_evaluation_metrics[strategy_tuple].get('reliability_per_level', {})
                            if fault_level in rel_per_level:
                                reliabilities.append(rel_per_level[fault_level])
            
            if reliabilities:
                multi_level_stats[f'reliability_{fault_level}_faults_mean'] = np.mean(reliabilities)
                multi_level_stats[f'reliability_{fault_level}_faults_std'] = np.std(reliabilities)
                multi_level_stats[f'reliability_{fault_level}_faults_max'] = np.max(reliabilities)
                multi_level_stats[f'reliability_{fault_level}_faults_min'] = np.min(reliabilities)
        
        return multi_level_stats
    
    def _detect_cliff_collapse(self, multi_level_stats: dict) -> dict:
     
        fault_levels = [10, 25, 50, 100, 150]
        reliabilities = []
        
        # Extract reliability curve
        for fault_level in fault_levels:
            rel_key = f'reliability_{fault_level}_faults_mean'
            if rel_key in multi_level_stats:
                reliabilities.append(multi_level_stats[rel_key])
            else:
                reliabilities.append(0.0)
        
        if len(reliabilities) < 3 or max(reliabilities) < 10.0:
            return None  # Not enough data or too low reliability
        
        # Find maximum single drop
        max_drop = 0.0
        cliff_location = -1
        cliff_from = 0.0
        cliff_to = 0.0
        
        for i in range(len(reliabilities) - 1):
            if reliabilities[i] > 0 and reliabilities[i+1] >= 0:
                drop = reliabilities[i] - reliabilities[i+1]
                if drop > max_drop:
                    max_drop = drop
                    cliff_location = i
                    cliff_from = reliabilities[i]
                    cliff_to = reliabilities[i+1]
        
        # Detect significant cliff (like your 48.676 → 14.502 = 34.17% drop)
        if max_drop > 20.0:  # Significant cliff threshold
            return {
                'max_drop': max_drop,
                'from_fault_level': fault_levels[cliff_location],
                'to_fault_level': fault_levels[cliff_location + 1],
                'from_reliability': cliff_from,
                'to_reliability': cliff_to,
                'cliff_severity': 'severe' if max_drop > 30.0 else 'moderate'
            }
        
        return None
    
    def _calculate_brittleness_score(self, multi_level_stats: dict) -> float:
       
        fault_levels = [10, 25, 50, 100, 150]
        reliabilities = []
        
        for fault_level in fault_levels:
            rel_key = f'reliability_{fault_level}_faults_mean'
            if rel_key in multi_level_stats:
                reliabilities.append(multi_level_stats[rel_key])
        
        if len(reliabilities) < 3:
            return 0.5  # Neutral if insufficient data
        
        valid_rels = [r for r in reliabilities if r > 0]
        if len(valid_rels) < 2:
            return 0.5
        
        brittleness = 0.0
        
        # Factor 1: Maximum single drop (cliff collapse)
        max_drop = 0.0
        for i in range(len(reliabilities) - 1):
            if reliabilities[i] > 0 and reliabilities[i+1] >= 0:
                drop = reliabilities[i] - reliabilities[i+1]
                max_drop = max(max_drop, drop)
        
        if max_drop > 25.0:
            brittleness += 0.4  # High brittleness for severe cliffs
        elif max_drop > 15.0:
            brittleness += 0.2
        
        # Factor 2: Consistency across fault levels
        consistency = min(valid_rels) / max(valid_rels) if max(valid_rels) > 0 else 0
        if consistency < 0.3:
            brittleness += 0.3  # High brittleness for low consistency
        elif consistency < 0.5:
            brittleness += 0.2
        
        # Factor 3: Early degradation (10 → 25 fault performance)
        if len(reliabilities) >= 2 and reliabilities[0] > 10.0:
            early_degradation = 1.0 - (reliabilities[1] / reliabilities[0])
            if early_degradation > 0.4:  # >40% drop early
                brittleness += 0.3
        
        return min(1.0, brittleness)
    
    def _inject_enhanced_diversity(self) -> None:
        """Enhanced diversity injection with adaptive exploration (RL disabled)."""
        # Always use sensitivity-aware diversity since RL is disabled
        self._inject_sensitivity_aware_diversity()
            
    def _inject_rl_guided_diversity(self) -> None:
        """RL-guided intelligent diversity injection - learns from previous attempts."""
        replace_count = int(len(self.population) * 0.4)
        self.diversity_injection_count += 1
        
        print(f"   RL-GUIDED diversity injection: replacing {replace_count} strategies (injection #{self.diversity_injection_count})")
        
        # Sort population to keep best individuals
        if hasattr(self.population[0], 'fitness') and self.population[0].fitness.valid:
            def safe_sort_key(ind):
                if ind.fitness.valid and len(ind.fitness.values) >= 2:
                    return (ind.fitness.values[1], ind.fitness.values[0])
                else:
                    return (-1000, -1000)
            self.population.sort(key=safe_sort_key, reverse=True)
        
        # Get RL learning stats to guide strategy generation
        rl_stats = self.rl_controller.get_learning_stats()
        global_best = rl_stats.get('global_best_reliability', 0.0)
        moving_avg = rl_stats.get('moving_window_avg', 0.0)
        adaptive_threshold = self.rl_controller.get_adaptive_penalty_threshold()
        
        # Fixed sparsity threshold context
        sparsity_target = self.ga_config.min_overall_sparsity_threshold  # Fixed at 55%
        
        print(f"   RL Learning: Best={global_best:.1f}%, MovingAvg={moving_avg:.1f}%, SparsityTarget={sparsity_target:.1f}%")
        
        # RL-guided strategy generation with FIXED 55% sparsity threshold
        for i in range(replace_count):
            replace_idx = len(self.population) - 1 - i
            if replace_idx >= 0:
                # Generate strategies based on RL learning + fixed 55% minimum sparsity
                min_sparsity = sparsity_target  # Fixed at 55%
                max_sparsity = 90.0  # Reasonable upper bound for exploration
                
                if global_best < 30.0:
                    # Very low performance: conservative exploration near 55%
                    if i % 3 == 0:
                        new_individual = self.population_initializer.uniform_random_individual(55.0, 65.0)  # Conservative
                    elif i % 3 == 1:
                        new_individual = self.population_initializer.uniform_random_individual(60.0, 70.0)  # Moderate
                    else:
                        new_individual = self.population_initializer.uniform_random_individual(55.0, 62.0)  # Very conservative
                        
                elif global_best < 50.0:
                    # Moderate performance: balanced exploration 
                    if i % 4 == 0:
                        # Around moving average ± 10%
                        center = max(55.0, min(85.0, moving_avg if moving_avg > 0 else 65.0))
                        new_individual = self.population_initializer.uniform_random_individual(center-10, center+10)
                    elif i % 4 == 1:
                        new_individual = self.population_initializer.uniform_random_individual(55.0, 70.0)  # Conservative
                    elif i % 4 == 2:
                        new_individual = self.population_initializer.uniform_random_individual(70.0, 85.0)  # Aggressive push
                    else:
                        new_individual = self.population_initializer.uniform_random_individual(60.0, 75.0)  # Moderate
                        
                else:
                    # Good performance: ambitious exploration for high sparsity + reliability
                    if i % 5 == 0:
                        # Push toward high sparsity for competitive compression
                        new_individual = self.population_initializer.uniform_random_individual(75.0, max_sparsity)
                    elif i % 5 == 1:
                        # Around adaptive threshold region
                        new_individual = self.population_initializer.uniform_random_individual(adaptive_threshold-10, adaptive_threshold+10)
                    elif i % 5 == 2:
                        new_individual = self.population_initializer.uniform_random_individual(55.0, 70.0)  # Conservative backup
                    elif i % 5 == 3:
                        new_individual = self.population_initializer.uniform_random_individual(65.0, 80.0)  # Moderate-high
                    else:
                        # Research-competitive exploration
                        new_individual = self.population_initializer.uniform_random_individual(80.0, max_sparsity)
                
                # Replace the individual
                self.population[replace_idx] = self.toolbox.individual_from_list(new_individual)
                self.population[replace_idx].fitness.delValues()
        
        print(f"   RL-guided diversity injection complete - strategies based on learned patterns")
        
    def _inject_random_diversity(self) -> None:
        """REPLACED: Use sensitivity-aware diversity injection instead of pure random."""
        self._inject_sensitivity_aware_diversity()

    def _inject_sensitivity_aware_diversity(self) -> None:
        """Sensitivity-aware diversity injection using high-reliability patterns."""
        replace_count = int(len(self.population) * 0.4)
        self.diversity_injection_count += 1
        
        print(f"   SENSITIVITY-AWARE diversity injection: replacing {replace_count} strategies (injection #{self.diversity_injection_count})")
        
        # Sort population to keep best individuals - with safety check
        if hasattr(self.population[0], 'fitness') and self.population[0].fitness.valid:
            def safe_sort_key(ind):
                if ind.fitness.valid and len(ind.fitness.values) >= 2:
                    return (ind.fitness.values[1], ind.fitness.values[0])
                else:
                    return (-1000, -1000)
            self.population.sort(key=safe_sort_key, reverse=True)
        
        # Extract high-reliability patterns from best individuals
        high_reliability_patterns = []
        for ind in self.population[:3]:  # Top 3 individuals
            if ind.fitness.valid and len(ind.fitness.values) >= 2 and ind.fitness.values[1] > 30.0:
                high_reliability_patterns.append(list(ind))
        
        # Calculate current population reliability stats
        reliabilities = [ind.fitness.values[1] for ind in self.population if ind.fitness.valid and len(ind.fitness.values) >= 2]
        target_reliability = np.mean(reliabilities) + 5.0 if reliabilities else 40.0
        
        # Generate sensitivity-informed diversity individuals
        new_individuals = self.population_initializer.create_reliability_guided_diversity(
            count=replace_count,
            high_reliability_patterns=high_reliability_patterns,
            target_reliability=target_reliability
        )
        
        # Replace worst individuals with sensitivity-aware strategies
        for i, new_individual in enumerate(new_individuals):
            replace_idx = len(self.population) - 1 - i
            if replace_idx >= 0:
                self.population[replace_idx] = self.toolbox.individual_from_list(new_individual)
                self.population[replace_idx].fitness.delValues()
        
        print(f"   Sensitivity-aware diversity injection complete - generated {len(new_individuals)} strategies guided by reliability patterns and layer sensitivity")
    
    def _inject_simple_diversity(self) -> None:
        """Simple diversity injection focused on reliability exploration."""
        # Replace worst 20% with diverse strategies
        replace_count = int(len(self.population) * 0.2)
        self.diversity_injection_count += 1
        
        print(f"   Injecting {replace_count} diverse strategies (injection #{self.diversity_injection_count})")
        
        # Sort population to keep best individuals - with safety check
        if hasattr(self.population[0], 'fitness') and self.population[0].fitness.valid:
            def safe_sort_key(ind):
                if ind.fitness.valid and len(ind.fitness.values) >= 2:
                    return (ind.fitness.values[1], ind.fitness.values[0])
                else:
                    return (-1000, -1000)  # Put invalid fitness at end
            self.population.sort(key=safe_sort_key, reverse=True)
        
        # Replace worst individuals with diverse strategies
        for i in range(replace_count):
            replace_idx = len(self.population) - 1 - i
            if replace_idx >= 0:
                # Create diverse strategies within research-relevant compression range
                if i % 3 == 0:
                    # Moderate compression (better reliability potential)
                    new_individual = self.population_initializer.uniform_random_individual(50.0, 65.0)
                elif i % 3 == 1:
                    # High compression - FIXED SAFE RANGE
                    new_individual = self.population_initializer.uniform_random_individual(65.0, 70.0)
                else:
                    # Aggressive compression (research frontier) - FIXED SAFE RANGE
                    new_individual = self.population_initializer.uniform_random_individual(70.0, 75.0)
                
                # Replace the individual
                self.population[replace_idx] = self.toolbox.individual_from_list(new_individual)
                self.population[replace_idx].fitness.delValues()
        
        print(f"   Diversity injection complete - added moderate (50-65%), high (65-80%), and aggressive (80-95%) compression strategies")
    
    def _check_stagnation_and_restart(self, generation: int) -> bool:
      
        if len(self.generation_stats) < 6:
            return False  # Start checking much earlier (6 instead of 10)
        
        # Check reliability progress over last 6 generations (more sensitive)
        recent_stats = self.generation_stats[-6:]
        reliabilities = [stat.get('reliability_mean', 0) for stat in recent_stats]
        
        if not reliabilities:
            return False
        
        # More sensitive stagnation criteria
        max_reliability = max(reliabilities)
        recent_improvement = reliabilities[-1] - reliabilities[0]  # 6-generation improvement
        
        # MULTIPLE stagnation triggers (any one triggers restart)
        reliability_stuck = recent_improvement < 1.0  # Less than 1% improvement over 6 gens
        low_absolute = max_reliability < 35.0  # Below 35% reliability  
        variance_low = np.var(reliabilities) < 2.0  # Population converged (low variance)
        late_enough = generation > 15  # Reduced from 20
        
        if (reliability_stuck and low_absolute and late_enough) or variance_low:
            print(f"STAGNATION: improvement={recent_improvement:.1f}%, max={max_reliability:.1f}%, var={np.var(reliabilities):.1f}")
            print("TRIGGERING POPULATION RESTART with elite preservation")
            
            # AGGRESSIVE RESTART: Keep only top 10% elite (instead of 20%)
            if hasattr(self.population[0], 'fitness') and self.population[0].fitness.valid:
                def safe_sort_key(ind):
                    if ind.fitness.valid and len(ind.fitness.values) >= 2:
                        return ind.fitness.values[1]  # Sort by reliability fitness
                    else:
                        return -1000  # Put invalid fitness at end
                self.population.sort(key=safe_sort_key, reverse=True)
                elite_count = int(len(self.population) * 0.1)  # Only 10% elite
                elite_individuals = self.population[:elite_count]
                
                # ADAPTIVE RESTART: Generate new population using unexplored strategies
                new_population_size = len(self.population) - elite_count
                new_individuals = self._generate_adaptive_restart_population(new_population_size)
                
                # Safety check: ensure new_individuals is not None
                if new_individuals is None:
                    print("WARNING: new_individuals is None, generating fallback population")
                    new_individuals = [self.population_initializer.uniform_random_individual(20.0, 95.0) 
                                     for _ in range(new_population_size)]
                
                # Replace non-elite with adaptive restart individuals
                for i, new_individual in enumerate(new_individuals):
                    if elite_count + i < len(self.population):
                        self.population[elite_count + i] = self.toolbox.individual_from_list(new_individual)
                        # Properly invalidate fitness instead of deleting values
                        self.population[elite_count + i].fitness.delValues()
                
                self.restart_count += 1
                print(f"ADAPTIVE RESTART #{self.restart_count}: {elite_count} elite preserved, {len(new_individuals)} NEW STRATEGY individuals")
                return True
        
        return False
    
    def _generate_adaptive_restart_population(self, population_size: int) -> List[List[float]]:
       
        # Analyze current population to understand what we've tried
        self._analyze_explored_regions()
        
        # Select restart strategy based on restart count and explored regions
        restart_strategies = self._determine_restart_strategies()
        
        print(f"RESTART STRATEGY: {restart_strategies['description']}")
        
        new_population = []
        
        # Generate individuals using the selected restart strategy
        for i in range(population_size):
            strategy_type = i % len(restart_strategies['generators'])
            generator_info = restart_strategies['generators'][strategy_type]
            
            if generator_info['type'] == 'extreme_range':
                # Use completely different sparsity ranges
                individual = self.population_initializer.uniform_random_individual(
                    generator_info['min_range'], generator_info['max_range']
                )
            elif generator_info['type'] == 'anti_pattern':
                # Generate anti-patterns to current population
                individual = self._generate_anti_pattern_individual()
            elif generator_info['type'] == 'sensitivity_inverse':
                # Use inverse sensitivity guidance
                individual = self._generate_sensitivity_inverse_individual()
            elif generator_info['type'] == 'ultra_conservative':
                individual = self.population_initializer._create_conservative_individual()
            elif generator_info['type'] == 'ultra_aggressive':
                individual = self.population_initializer._create_ultra_aggressive_individual()
            else:
                # Fallback: random extreme
                individual = self.population_initializer.uniform_random_individual(20.0, 95.0)
            
            # Safety check: ensure individual is not None
            if individual is None:
                individual = self.population_initializer.uniform_random_individual(20.0, 95.0)
                
            new_population.append(individual)
        
        return new_population
    
    def _analyze_explored_regions(self):
        """Analyze current population to understand explored sparsity regions."""
        if not hasattr(self.population[0], 'fitness') or not self.population[0].fitness.valid:
            return
        
        # Calculate average sparsity of current population
        population_sparsities = []
        for individual in self.population:
            if self.population_initializer.global_mode:
                sparsity = individual[0]  # Global sparsity value
            else:
                sparsity = sum(individual) / len(individual)  # Average layer sparsity
            population_sparsities.append(sparsity)
        
        avg_sparsity = sum(population_sparsities) / len(population_sparsities)
        self.explored_regions.append({
            'restart_count': self.restart_count,
            'avg_sparsity': avg_sparsity,
            'sparsity_range': (min(population_sparsities), max(population_sparsities))
        })
        
        print(f"Current population: Avg sparsity = {avg_sparsity:.1f}%, Range = {min(population_sparsities):.1f}%-{max(population_sparsities):.1f}%")
    
    def _determine_restart_strategies(self) -> Dict:
        """Determine which restart strategies to use based on history."""
        
        if self.restart_count == 0:
            # First restart: Try opposite extremes
            return {
                'description': "First restart - exploring opposite extremes",
                'generators': [
                    {'type': 'ultra_conservative', 'min_range': 25.0, 'max_range': 45.0},
                    {'type': 'ultra_aggressive', 'min_range': 70.0, 'max_range': 75.0},
                    {'type': 'extreme_range', 'min_range': 20.0, 'max_range': 50.0},
                    {'type': 'extreme_range', 'min_range': 70.0, 'max_range': 75.0}
                ]
            }
        
        elif self.restart_count == 1:
            # Second restart: Anti-patterns and sensitivity inverse
            return {
                'description': "Second restart - anti-patterns and inverse sensitivity",
                'generators': [
                    {'type': 'anti_pattern'},
                    {'type': 'sensitivity_inverse'},
                    {'type': 'extreme_range', 'min_range': 15.0, 'max_range': 35.0},
                    {'type': 'extreme_range', 'min_range': 85.0, 'max_range': 98.0}
                ]
            }
        
        else:
            # Third+ restart: Completely random exploration
            return {
                'description': f"Restart #{self.restart_count} - random exploration with wider bounds",
                'generators': [
                    {'type': 'extreme_range', 'min_range': 10.0, 'max_range': 30.0},
                    {'type': 'extreme_range', 'min_range': 30.0, 'max_range': 50.0},
                    {'type': 'extreme_range', 'min_range': 50.0, 'max_range': 70.0},
                    {'type': 'extreme_range', 'min_range': 70.0, 'max_range': 98.0}
                ]
            }
    
    def _generate_anti_pattern_individual(self) -> List[float]:
        """Generate individual that's opposite to current population patterns."""
        if not self.population or not hasattr(self.population[0], 'fitness'):
            return self.population_initializer.uniform_random_individual(20.0, 95.0)
        
        # Find the best individual and create its opposite
        def safe_fitness_key(ind):
            if ind.fitness.valid and len(ind.fitness.values) >= 2:
                return ind.fitness.values[1]
            else:
                return -1
        best_individual = max(self.population, key=safe_fitness_key)
        
        if self.population_initializer.global_mode:
            # For global mode, if best is high sparsity, try low sparsity and vice versa
            best_sparsity = best_individual[0]
            if best_sparsity > 60.0:
                anti_sparsity = random.uniform(25.0, 45.0)  # Go conservative
            else:
                anti_sparsity = random.uniform(75.0, 95.0)  # Go aggressive
            return [anti_sparsity]
        else:
            # For layer-wise mode, invert the pattern
            anti_individual = []
            for i, gene in enumerate(best_individual):
                if gene > 50.0:
                    anti_gene = random.uniform(10.0, 30.0)  # High -> Low
                else:
                    anti_gene = random.uniform(70.0, 90.0)  # Low -> High
                anti_individual.append(anti_gene)
            return anti_individual
    
    def _generate_sensitivity_inverse_individual(self) -> List[float]:
        """Generate individual using inverse sensitivity logic."""
        # This is a placeholder - could be enhanced with actual inverse sensitivity logic
        return self.population_initializer.uniform_random_individual(15.0, 85.0)
    
    def _apply_meta_learning_adaptations(self, generation: int):
        """Apply autonomous meta-learning adaptations based on progress analysis."""
        # Disabled - meta_controller functionality removed
        print(f"Meta-learning adaptations disabled for generation {generation}")
        return
    
    def _is_too_similar(self, ind1, ind2) -> bool:
    
        # Calculate Euclidean distance between individuals
        distance = sum((a - b) ** 2 for a, b in zip(ind1, ind2)) ** 0.5
        
        # Minimum distance threshold for diversity
        if self.ga_config.global_mode:
            min_distance = 2.0  # Smaller threshold for global mode (1 parameter)
        else:
            min_distance = 5.0  # Larger threshold for layer-wise mode (19 parameters)
        
        return distance < min_distance
    
    def _print_generation_summary(self, generation: int):
        """Print summary of current generation."""
        if not self.generation_stats:
            return
        
        stats = self.generation_stats[-1]
        print(f"Gen {generation:3d}: "
              f"Acc={stats.get('accuracy_mean', 0):.2f}+/-{stats.get('accuracy_std', 0):.2f}, "
              f"Rel={stats.get('reliability_mean', 0):.2f}+/-{stats.get('reliability_std', 0):.2f}, "
              f"Div={stats.get('population_diversity', 0):.1f}, "
              f"HoF={len(self.hall_of_fame)}")
    
    # Checkpoint methods removed to avoid pickle multiprocessing errors
    
    def _save_final_results(self):
        """Save final evolution results."""
        import os
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure ga_results directory exists
        results_dir = self.ga_config.results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Determine filename prefix based on mode
        mode_prefix = "Global_" if self.ga_config.global_mode else "Layerwise_"
        
        # Save Pareto front (ALWAYS create file, even if empty)
        pareto_filename = os.path.join(results_dir, f"{mode_prefix}pareto_front_solutions_{timestamp}.csv")
        self._save_pareto_front_csv(pareto_filename)
        
        # Save evolution statistics
        stats_filename = os.path.join(results_dir, f"{mode_prefix}evolution_stats_{timestamp}.csv")
        if self.generation_stats:
            df = pd.DataFrame(self.generation_stats)
            df.to_csv(stats_filename, index=False)
            print(f"Evolution statistics saved to {stats_filename}")
        
        # Save all evaluation results
        results_filename = os.path.join(results_dir, f"{mode_prefix}all_evaluations_{timestamp}.csv")
        
        if self.all_evaluation_results and len(self.all_evaluation_results) > 0:
            flattened_results = []
            for result in self.all_evaluation_results:
                row = {
                    'generation': result['generation'],
                    'individual_id': result['individual_id'],
                    'accuracy_fitness': result['fitness'][0],
                    'reliability_fitness': result['fitness'][1],
                }
                # Add strategy parameters
                for i, param in enumerate(result['strategy']):
                    row[f'param_{i}'] = param
                flattened_results.append(row)
            
            df = pd.DataFrame(flattened_results)
        else:
            # Create empty DataFrame with standard headers
            df = pd.DataFrame(columns=['generation', 'individual_id', 'accuracy_fitness', 'reliability_fitness'])
        
        try:
            df.to_csv(results_filename, index=False)
            print(f"All evaluation results saved to {results_filename}")
        except Exception as e:
            print(f"Error saving evaluation results: {e}")
    
    def _save_pareto_front_csv(self, filename: str):
        """Save Pareto front solutions to CSV (matching old code format)."""
        
        if not self.hall_of_fame or len(self.hall_of_fame) == 0:
            self._create_empty_pareto_csv(filename)
            return
        
        # Get layer names for headers - CRITICAL: Use sensitivity ordering!
        score_files_map = get_layer_score_files_map(
            self.ga_config.score_dir_path,
            self.base_model.state_dict()
        )
        available_layers = [
            name for name in self.base_model.state_dict().keys()
            if ('features' in name or 'classifier' in name) and 'weight' in name
            and name in score_files_map
        ]
        prunable_layers = get_sensitivity_based_layer_ordering(available_layers, self.ga_config.score_dir_path)

        # Create headers matching old code format
        header_cols = ["Actual_Accuracy", "Actual_Sparsity", "Latency_ms", "Estimated_Reliability"]
        header_cols.extend([f"Layer_{layer.replace('.', '_')}_Gene" for layer in prunable_layers])
        
        
        # Write CSV matching old code format
        try:
            with open(filename, 'w') as f:
                f.write(','.join(header_cols) + '\n')
                
                rows_written = 0
                # Use direct cache lookup from fitness evaluator (same keys as evaluation)
                for individual in self.hall_of_fame:
                    if hasattr(individual, 'fitness') and individual.fitness.valid:
                        # Use same key format as evaluation.py stores
                        if self.ga_config.global_mode:
                            strategy_key = f"global_{individual[0]:.3f}"
                        else:
                            strategy_key = tuple(round(x, 2) for x in individual)
                        # Get actual metrics from fitness evaluator cache
                        actual_metrics = self.fitness_evaluator.last_evaluation_metrics.get(strategy_key)
                        
                        if actual_metrics:
                            # Use actual measured metrics from cache
                            actual_acc_val = actual_metrics.get("accuracy", 0)
                            actual_sp_val = actual_metrics.get("sparsity", 0) 
                            actual_lat_val = actual_metrics.get("latency", 0)
                            actual_rel_val = actual_metrics.get("reliability", 0)
                            applied_percentages = actual_metrics.get("applied_layer_percentages", None)
                        else:
                            # FALLBACK: Search all_evaluation_results for matching strategy
                            actual_acc_val = actual_sp_val = actual_lat_val = actual_rel_val = 0
                            applied_percentages = None
                            
                            strategy_tuple = tuple(individual)
                            for result in self.all_evaluation_results:
                                result_tuple = tuple(result['strategy'])
                                if result_tuple == strategy_tuple:
                                    actual_acc_val = result.get('accuracy', 0)
                                    actual_sp_val = result.get('sparsity', 0)
                                    actual_lat_val = result.get('latency', 0)
                                    actual_rel_val = result.get('reliability', 0)
                                    applied_percentages = result.get('applied_layer_percentages', None)
                                    break
                        
                        # Handle parameter formatting
                        if self.ga_config.global_mode and len(individual) == 1:
                            # For global mode: use pre-stored applied layer percentages if available
                            if applied_percentages and len(applied_percentages) == len(prunable_layers):
                                param_vals_str = ','.join([f"{p:.4f}" for p in applied_percentages])
                            else:
                                param_vals_str = ','.join([f"{individual[0]:.4f}" for _ in prunable_layers])
                        else:
                            # For layer-wise mode: use evolved parameters as-is
                            param_vals_str = ','.join([f"{p:.4f}" for p in individual])
                        
                        f.write(f"{actual_acc_val:.4f},{actual_sp_val:.4f},{actual_lat_val:.4f},{actual_rel_val:.4f},{param_vals_str}\n")
                        rows_written += 1
            
            print(f"Saved Pareto front solutions to {filename}")
            
        except Exception as e:
            print(f"Error saving Pareto front CSV: {e}")
    
    def _create_empty_pareto_csv(self, filename: str):
        """Create empty Pareto CSV with headers for debugging when hall_of_fame is empty."""
        
        # Get layer names for headers - CRITICAL: Use sensitivity ordering!
        score_files_map = get_layer_score_files_map(
            self.ga_config.score_dir_path,
            self.base_model.state_dict()
        )
        available_layers = [
            name for name in self.base_model.state_dict().keys()
            if ('features' in name or 'classifier' in name) and 'weight' in name
            and name in score_files_map
        ]
        prunable_layers = get_sensitivity_based_layer_ordering(available_layers, self.ga_config.score_dir_path)

        # Create headers matching normal CSV format
        header_cols = ["Actual_Accuracy", "Actual_Sparsity", "Latency_ms", "Estimated_Reliability"]
        header_cols.extend([f"Layer_{layer.replace('.', '_')}_Gene" for layer in prunable_layers])
        
        # Write empty CSV with headers only
        with open(filename, 'w') as f:
            f.write(','.join(header_cols) + '\n')
            # No data rows - indicates empty hall of fame
        
        print(f"Created empty Pareto CSV: {filename}")
        print(f"Check GA debug output above to see why individuals had invalid fitness.")
    
    def _create_metrics_map(self) -> Dict:
        """Create mapping from strategies to detailed metrics."""
        metrics_map = {}
        
        for result in self.all_evaluation_results:
            strategy_tuple = tuple(result['strategy'])
            if strategy_tuple not in metrics_map:
                metrics_map[strategy_tuple] = {
                    'accuracy_fitness': result['fitness'][0],
                    'reliability_fitness': result['fitness'][1],
                    'generation': result['generation']
                }
        
        return metrics_map
    
    def _create_detailed_metrics_map(self) -> Dict:
        """Create mapping from strategies to actual measured metrics (not fitness values)."""
        metrics_map = {}
        
        for result in self.all_evaluation_results:
            strategy_tuple = tuple(result['strategy'])
            if strategy_tuple not in metrics_map:
                # Extract actual metrics from the evaluation results
                metrics_map[strategy_tuple] = {
                    'accuracy': result.get('accuracy', 0),
                    'sparsity': result.get('sparsity', 0),
                    'latency': result.get('latency', 0), 
                    'reliability': result.get('reliability', 0),
                    'generation': result['generation'],
                    'applied_layer_percentages': result.get('applied_layer_percentages', None)
                }
        
        return metrics_map
    
    def get_best_solutions(self, n: int = 5) -> List[Dict]:
       
        if not self.hall_of_fame:
            return []
        
        # Sort by reliability first, then accuracy, then sparsity
        sorted_solutions = sorted(
            self.hall_of_fame,
            key=lambda x: (x.fitness.values[1], x.fitness.values[0]),  # reliability, accuracy
            reverse=True
        )
        
        best_solutions = []
        for i, individual in enumerate(sorted_solutions[:n]):
            solution = {
                'rank': i + 1,
                'strategy': individual[:],
                'fitness': individual.fitness.values,
                'accuracy_fitness': individual.fitness.values[0],
                'reliability_fitness': individual.fitness.values[1],
            }
            best_solutions.append(solution)
        
        return best_solutions
    
    def print_final_summary(self):
        """Print final evolution summary."""
        print(f"\n{'='*80}")
        print(f"FINAL EVOLUTION SUMMARY")
        print(f"{'='*80}")
        
        print(f"Generations completed: {len(self.generation_stats)}")
        print(f"Total evaluations: {len(self.all_evaluation_results)}")
        print(f"Pareto front size: {len(self.hall_of_fame)}")
        
        if self.hall_of_fame:
            print(f"\nTop 5 Solutions (by Reliability):")
            best_solutions = self.get_best_solutions(5)
            
            for solution in best_solutions:
                print(f"  Rank {solution['rank']}: "
                      f"Acc={solution['accuracy_fitness']:.2f}, "
                      f"Rel={solution['reliability_fitness']:.2f}")
        
        print(f"{'='*80}")
    
    def _create_extreme_layer_focused_individual(self) -> List[float]:
        """Create individual with extreme layer-wise differences for maximum diversity."""
        individual = []
        num_layers = len(self.population[0]) if self.population else 8
        
        for i in range(num_layers):
            if i % 3 == 0:
                # Very conservative layers
                individual.append(random.uniform(10.0, 25.0))
            elif i % 3 == 1:
                # Very aggressive layers
                individual.append(random.uniform(85.0, 95.0))
            else:
                # Moderate layers
                individual.append(random.uniform(45.0, 65.0))
        
        return individual
    
    def _emergency_population_restart(self) -> None:
        """
        Emergency population restart when complete convergence is detected.
        Keeps only top 3 individuals and regenerates rest with extreme diversity.
        """
        print("EMERGENCY RESTART: Complete population regeneration")
        
        # Keep only top 3 individuals
        if hasattr(self.population[0], 'fitness') and self.population[0].fitness.valid:
            self.population.sort(key=lambda ind: (ind.fitness.values[1], ind.fitness.values[0]), reverse=True)
            elite_individuals = self.population[:3]
        else:
            elite_individuals = self.population[:3]
        
        # Generate completely new population with EXTREME diversity
        new_population = []
        population_size = len(self.population)
        
        # Add back elite
        new_population.extend(elite_individuals)
        
        # Generate rest with guaranteed extreme diversity
        strategies = [
            (10.0, 20.0),   # Ultra-conservative
            (20.0, 35.0),   # Conservative  
            (35.0, 50.0),   # Moderate-low
            (50.0, 65.0),   # Moderate-high
            (65.0, 70.0),   # Aggressive (FIXED)
            (70.0, 75.0),   # Ultra-aggressive (CAPPED)
        ]
        
        for i in range(3, population_size):
            strategy_idx = i % len(strategies)
            min_val, max_val = strategies[strategy_idx]
            
            # Create individual in this range with additional noise
            new_individual = self.population_initializer.uniform_random_individual(min_val, max_val)
            
            # Add layer-wise variation to prevent identical strategies
            for j in range(len(new_individual)):
                layer_noise = random.uniform(-5.0, 5.0)
                new_individual[j] = max(min_val, min(max_val, new_individual[j] + layer_noise))
            
            new_population.append(self.toolbox.individual_from_list(new_individual))
            # Mark fitness as invalid
            new_population[-1].fitness.delValues()
        
        self.population[:] = new_population
        print(f"EMERGENCY RESTART COMPLETE: Generated {len(new_population)-3} new individuals")


def run_reliability_aware_ga(ga_config: GAConfig, model_config) -> ReliabilityAwareGA:
   
    ga_runner = ReliabilityAwareGA(ga_config, model_config)
    population, logbook, hof, metrics_map = ga_runner.run_evolution()
    ga_runner.print_final_summary()
    
    return ga_runner