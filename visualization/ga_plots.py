"""GA evolution visualization and trend analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import os
from datetime import datetime


class GAPlotter:
    """Visualization suite for GA evolution tracking and analysis."""
    
    def __init__(self, save_plots: bool = True, show_plots: bool = False):
        """
        Initialize GA plotter.
        
        Args:
            save_plots: Whether to save plots to files
            show_plots: Whether to display plots interactively
        """
        self.save_plots = save_plots
        self.show_plots = show_plots
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create results directory
        self.plots_dir = Path("ga_plots")
        self.plots_dir.mkdir(exist_ok=True)
        
    def plot_evolution_trends(self, generation_stats: List[Dict], 
                             title_prefix: str = "GA Evolution") -> None:
        """
        Plot generation-wise evolution trends for all objectives.
        
        Args:
            generation_stats: List of generation statistics dictionaries
            title_prefix: Prefix for plot titles
        """
        if not generation_stats:
            print("No generation statistics available for plotting")
            return
            
        df = pd.DataFrame(generation_stats)
        
        # Create comprehensive evolution plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{title_prefix} - Evolution Trends', fontsize=16, fontweight='bold')
        
        # 1. Accuracy Evolution
        if 'accuracy_mean' in df.columns:
            ax = axes[0, 0]
            ax.plot(df['generation'], df['accuracy_mean'], 'b-', linewidth=2, label='Mean')
            ax.fill_between(df['generation'], 
                           df['accuracy_mean'] - df['accuracy_std'], 
                           df['accuracy_mean'] + df['accuracy_std'], 
                           alpha=0.3, color='blue')
            ax.plot(df['generation'], df['accuracy_max'], 'g--', alpha=0.7, label='Max')
            ax.plot(df['generation'], df['accuracy_min'], 'r--', alpha=0.7, label='Min')
            ax.set_title('Accuracy Evolution', fontweight='bold')
            ax.set_ylabel('Accuracy (%)')
            ax.set_xlabel('Generation')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Reliability Evolution (KEY PLOT)
        if 'reliability_mean' in df.columns:
            ax = axes[0, 1]
            ax.plot(df['generation'], df['reliability_mean'], 'r-', linewidth=3, label='Mean Reliability')
            ax.fill_between(df['generation'], 
                           df['reliability_mean'] - df['reliability_std'], 
                           df['reliability_mean'] + df['reliability_std'], 
                           alpha=0.3, color='red')
            ax.plot(df['generation'], df['reliability_max'], 'orange', linewidth=2, label='Best Reliability')
            ax.plot(df['generation'], df['reliability_min'], 'darkred', alpha=0.7, label='Worst Reliability')
            ax.set_title(' RELIABILITY MONSTER EVOLUTION ', fontweight='bold', color='red')
            ax.set_ylabel('Actual Reliability (%)')
            ax.set_xlabel('Generation')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add reliability quality zones
            ax.axhspan(90, 100, alpha=0.1, color='green', label='LEGENDARY')
            ax.axhspan(70, 90, alpha=0.1, color='blue', label='EXCELLENT') 
            ax.axhspan(50, 70, alpha=0.1, color='orange', label='GOOD')
            ax.axhspan(0, 50, alpha=0.1, color='red', label='POOR')
        
        # 3. Population Diversity
        if 'population_diversity' in df.columns:
            ax = axes[0, 2]
            ax.plot(df['generation'], df['population_diversity'], 'm-', linewidth=2)
            ax.set_title('Population Diversity', fontweight='bold')
            ax.set_ylabel('Diversity Score')
            ax.set_xlabel('Generation')
            ax.grid(True, alpha=0.3)
            
            # Add danger zone for low diversity
            ax.axhline(y=5.0, color='red', linestyle='--', alpha=0.7, label='Danger Zone')
            ax.legend()
        
        # 4. Hall of Fame Growth
        if 'hall_of_fame_size' in df.columns:
            ax = axes[1, 0]
            ax.plot(df['generation'], df['hall_of_fame_size'], 'g-', linewidth=2, marker='o')
            ax.set_title('Pareto Front Growth', fontweight='bold')
            ax.set_ylabel('Pareto Front Size')
            ax.set_xlabel('Generation')
            ax.grid(True, alpha=0.3)
        
        # 5. Multi-objective Progress
        if all(col in df.columns for col in ['accuracy_mean', 'reliability_mean']):
            ax = axes[1, 1]
            # Normalize values for comparison
            acc_norm = (df['accuracy_mean'] - df['accuracy_mean'].min()) / (df['accuracy_mean'].max() - df['accuracy_mean'].min())
            rel_norm = (df['reliability_mean'] - df['reliability_mean'].min()) / (df['reliability_mean'].max() - df['reliability_mean'].min())
            
            ax.plot(df['generation'], acc_norm, 'b-', linewidth=2, label='Accuracy (normalized)')
            ax.plot(df['generation'], rel_norm, 'r-', linewidth=2, label='Reliability (normalized)')
            ax.set_title('Normalized Objectives Progress', fontweight='bold')
            ax.set_ylabel('Normalized Progress (0-1)')
            ax.set_xlabel('Generation')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 6. Reliability Quality Distribution  
        if 'reliability_mean' in df.columns:
            ax = axes[1, 2]
            final_gen = df.iloc[-5:] if len(df) >= 5 else df  # Last 5 generations
            reliability_values = []
            for _, row in final_gen.iterrows():
                # Approximate distribution from mean/std
                mean_rel = row['reliability_mean']
                std_rel = row.get('reliability_std', 5.0)
                reliability_values.extend(np.random.normal(mean_rel, std_rel, 20))
            
            ax.hist(reliability_values, bins=20, alpha=0.7, color='red', edgecolor='black')
            ax.set_title('Recent Reliability Distribution', fontweight='bold')
            ax.set_xlabel('Reliability (%)')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.plots_dir / f"evolution_trends_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Evolution trends plot saved to {filename}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_fitness_vs_actual_reliability(self, all_evaluations: List[Dict],
                                         title: str = "Fitness vs Actual Reliability") -> None:
        """
        Plot fitness scores vs actual reliability to detect scoring problems.
        
        Args:
            all_evaluations: List of all evaluation results
            title: Plot title
        """
        if not all_evaluations:
            print("No evaluation data available for fitness analysis")
            return
            
        # Extract data
        actual_reliability = []
        reliability_fitness = []
        accuracy_fitness = []
        generations = []
        
        for eval_data in all_evaluations:
            if 'reliability' in eval_data and 'fitness' in eval_data:
                actual_reliability.append(eval_data['reliability'])
                if len(eval_data['fitness']) >= 2:
                    accuracy_fitness.append(eval_data['fitness'][0])
                    reliability_fitness.append(eval_data['fitness'][1])
                    generations.append(eval_data.get('generation', 0))
        
        if not actual_reliability:
            print("No reliability data found in evaluations")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{title} - Scoring System Analysis', fontsize=16, fontweight='bold')
        
        # 1. Reliability Fitness vs Actual Reliability
        ax = axes[0, 0]
        scatter = ax.scatter(actual_reliability, reliability_fitness, 
                           c=generations, cmap='viridis', alpha=0.6, s=50)
        ax.set_xlabel('Actual Reliability (%)')
        ax.set_ylabel('Reliability Fitness Score')
        ax.set_title(' RELIABILITY MONSTER SCORING ', fontweight='bold', color='red')
        ax.grid(True, alpha=0.3)
        
        # Add trend line to check monotonicity
        if len(actual_reliability) > 1:
            z = np.polyfit(actual_reliability, reliability_fitness, 1)
            p = np.poly1d(z)
            ax.plot(sorted(actual_reliability), p(sorted(actual_reliability)), "r--", alpha=0.8, linewidth=2)
        
        # Add colorbar for generations
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Generation')
        
        # 2. Accuracy Fitness vs Actual Accuracy (if available)
        ax = axes[0, 1]
        if 'accuracy' in all_evaluations[0]:
            actual_accuracy = [eval_data.get('accuracy', 0) for eval_data in all_evaluations]
            ax.scatter(actual_accuracy, accuracy_fitness, 
                      c=generations, cmap='plasma', alpha=0.6, s=50)
            ax.set_xlabel('Actual Accuracy (%)')
            ax.set_ylabel('Accuracy Fitness Score')
            ax.set_title('Accuracy Scoring Check', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(actual_accuracy) > 1:
                z = np.polyfit(actual_accuracy, accuracy_fitness, 1)
                p = np.poly1d(z)
                ax.plot(sorted(actual_accuracy), p(sorted(actual_accuracy)), "b--", alpha=0.8, linewidth=2)
        
        # 3. Generation-wise Fitness Evolution
        ax = axes[1, 0]
        gen_groups = {}
        for i, gen in enumerate(generations):
            if gen not in gen_groups:
                gen_groups[gen] = {'rel_fitness': [], 'actual_rel': []}
            gen_groups[gen]['rel_fitness'].append(reliability_fitness[i])
            gen_groups[gen]['actual_rel'].append(actual_reliability[i])
        
        gen_numbers = sorted(gen_groups.keys())
        mean_rel_fitness = [np.mean(gen_groups[g]['rel_fitness']) for g in gen_numbers]
        mean_actual_rel = [np.mean(gen_groups[g]['actual_rel']) for g in gen_numbers]
        
        ax2 = ax.twinx()
        line1 = ax.plot(gen_numbers, mean_rel_fitness, 'r-', linewidth=2, label='Mean Reliability Fitness')
        line2 = ax2.plot(gen_numbers, mean_actual_rel, 'b-', linewidth=2, label='Mean Actual Reliability')
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Mean Reliability Fitness', color='red')
        ax2.set_ylabel('Mean Actual Reliability (%)', color='blue')
        ax.set_title('Evolution: Fitness vs Actual Performance', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        # 4. Problematic Cases Detection
        ax = axes[1, 1]
        
        # Find problematic cases: high fitness with low actual reliability
        problematic_indices = []
        for i in range(len(actual_reliability)):
            if reliability_fitness[i] > 500 and actual_reliability[i] < 60:
                problematic_indices.append(i)
        
        if problematic_indices:
            prob_rel = [actual_reliability[i] for i in problematic_indices]
            prob_fit = [reliability_fitness[i] for i in problematic_indices]
            prob_gen = [generations[i] for i in problematic_indices]
            
            ax.scatter(prob_rel, prob_fit, c='red', s=100, alpha=0.8, 
                      label=f'Problematic Cases ({len(problematic_indices)})')
            
            for i, (rel, fit, gen) in enumerate(zip(prob_rel, prob_fit, prob_gen)):
                ax.annotate(f'Gen {gen}', (rel, fit), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        # Plot all points for context
        ax.scatter(actual_reliability, reliability_fitness, alpha=0.3, c='gray', s=20)
        ax.set_xlabel('Actual Reliability (%)')
        ax.set_ylabel('Reliability Fitness Score')
        ax.set_title(' PROBLEMATIC SCORING DETECTION ', fontweight='bold', color='red')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if self.save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.plots_dir / f"fitness_analysis_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Fitness analysis plot saved to {filename}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_real_time_progress(self, generation_stats: List[Dict], 
                               current_gen: int) -> None:
        """
        Create real-time progress plot during GA evolution.
        
        Args:
            generation_stats: Generation statistics so far
            current_gen: Current generation number
        """
        if len(generation_stats) < 2:
            return
            
        # Create a simple real-time plot
        plt.figure(figsize=(12, 8))
        
        df = pd.DataFrame(generation_stats)
        
        # Plot key metrics
        plt.subplot(2, 2, 1)
        plt.plot(df['generation'], df['reliability_mean'], 'r-', linewidth=2)
        plt.fill_between(df['generation'], 
                        df['reliability_mean'] - df['reliability_std'], 
                        df['reliability_mean'] + df['reliability_std'], 
                        alpha=0.3, color='red')
        plt.title(f'Reliability Evolution (Gen {current_gen})', fontweight='bold')
        plt.ylabel('Reliability (%)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(df['generation'], df['accuracy_mean'], 'b-', linewidth=2)
        plt.fill_between(df['generation'], 
                        df['accuracy_mean'] - df['accuracy_std'], 
                        df['accuracy_mean'] + df['accuracy_std'], 
                        alpha=0.3, color='blue')
        plt.title(f'Accuracy Evolution (Gen {current_gen})', fontweight='bold')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(df['generation'], df['population_diversity'], 'm-', linewidth=2)
        plt.axhline(y=5.0, color='red', linestyle='--', alpha=0.7)
        plt.title(f'Diversity (Gen {current_gen})', fontweight='bold')
        plt.ylabel('Diversity Score')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.plot(df['generation'], df['hall_of_fame_size'], 'g-', linewidth=2, marker='o')
        plt.title(f'Pareto Front Size (Gen {current_gen})', fontweight='bold')
        plt.ylabel('Solutions')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            filename = self.plots_dir / f"realtime_gen_{current_gen:03d}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def create_reliability_monster_report(self, generation_stats: List[Dict], 
                                        all_evaluations: List[Dict],
                                        pareto_solutions: List = None) -> str:
        """
        Generate comprehensive reliability monster performance report.
        
        Args:
            generation_stats: Generation statistics
            all_evaluations: All evaluation results
            pareto_solutions: Final Pareto front solutions
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("" * 80)
        report.append("         GENIE RELIABILITY MONSTER PERFORMANCE REPORT")
        report.append("" * 80)
        
        if not generation_stats or not all_evaluations:
            report.append(" No data available for analysis")
            return "\n".join(report)
        
        # Evolution Summary
        df = pd.DataFrame(generation_stats)
        total_gens = len(df)
        final_gen = df.iloc[-1]
        
        report.append(f"\n EVOLUTION SUMMARY:")
        report.append(f"   Total Generations: {total_gens}")
        report.append(f"   Final Reliability Mean: {final_gen.get('reliability_mean', 0):.2f}% ± {final_gen.get('reliability_std', 0):.2f}%")
        report.append(f"   Final Accuracy Mean: {final_gen.get('accuracy_mean', 0):.2f}% ± {final_gen.get('accuracy_std', 0):.2f}%")
        report.append(f"   Final Population Diversity: {final_gen.get('population_diversity', 0):.2f}")
        report.append(f"   Pareto Front Size: {final_gen.get('hall_of_fame_size', 0)}")
        
        # Reliability Monster Analysis
        reliability_values = [eval_data.get('reliability', 0) for eval_data in all_evaluations if 'reliability' in eval_data]
        fitness_values = [eval_data['fitness'][1] for eval_data in all_evaluations if 'fitness' in eval_data and len(eval_data['fitness']) >= 2]
        
        if reliability_values:
            report.append(f"\n RELIABILITY MONSTER ACHIEVEMENTS:")
            
            legendary_count = len([r for r in reliability_values if r > 90])
            epic_count = len([r for r in reliability_values if 80 <= r <= 90])
            good_count = len([r for r in reliability_values if 60 <= r < 80])
            total_evals = len(reliability_values)
            
            report.append(f"    LEGENDARY (>90%): {legendary_count}/{total_evals} ({legendary_count/total_evals*100:.1f}%)")
            report.append(f"    EPIC (80-90%): {epic_count}/{total_evals} ({epic_count/total_evals*100:.1f}%)")
            report.append(f"    GOOD (60-80%): {good_count}/{total_evals} ({good_count/total_evals*100:.1f}%)")
            
            best_reliability = max(reliability_values)
            worst_reliability = min(reliability_values)
            avg_reliability = np.mean(reliability_values)
            
            report.append(f"   Best Reliability Achieved: {best_reliability:.2f}%")
            report.append(f"   Average Reliability: {avg_reliability:.2f}%")
            report.append(f"   Reliability Range: {worst_reliability:.2f}% - {best_reliability:.2f}%")
        
        # Scoring System Analysis
        if reliability_values and fitness_values and len(reliability_values) == len(fitness_values):
            report.append(f"\n️  SCORING SYSTEM ANALYSIS:")
            
            # Check for problematic cases
            problematic_cases = []
            for i, (rel, fit) in enumerate(zip(reliability_values, fitness_values)):
                if fit > 500 and rel < 60:
                    problematic_cases.append((rel, fit, i))
            
            if problematic_cases:
                report.append(f"    PROBLEMATIC SCORING DETECTED: {len(problematic_cases)} cases")
                report.append(f"   High fitness (>500) with low reliability (<60%)")
                
                for rel, fit, idx in problematic_cases[:5]:  # Show first 5
                    report.append(f"     Case {idx}: {rel:.1f}% reliability → {fit:.1f} fitness")
            else:
                report.append(f"    No obvious scoring problems detected")
            
            # Monotonicity check
            correlation = np.corrcoef(reliability_values, fitness_values)[0, 1]
            report.append(f"   Reliability-Fitness Correlation: {correlation:.3f}")
            
            if correlation < 0.7:
                report.append(f"   ️  WARNING: Low correlation suggests non-monotonic scoring!")
            else:
                report.append(f"    Good correlation between reliability and fitness")
        
        # Recommendations
        report.append(f"\n RECOMMENDATIONS:")
        
        if final_gen.get('population_diversity', 10) < 5.0:
            report.append(f"   ️  Low diversity detected - consider increasing mutation rate")
        
        if legendary_count / max(total_evals, 1) < 0.1:
            report.append(f"    Consider adjusting reliability thresholds to achieve more legendary solutions")
        
        if 'problematic_cases' in locals() and problematic_cases:
            report.append(f"    Fix exponential scoring system to ensure monotonic reliability rewards")
        
        report.append(f"\n" + "" * 80)
        
        return "\n".join(report)


def create_ga_plotter(save_plots: bool = True, show_plots: bool = False) -> GAPlotter:
    """
    Factory function to create GA plotter.
    
    Args:
        save_plots: Whether to save plots
        show_plots: Whether to display plots
        
    Returns:
        Configured GAPlotter instance
    """
    return GAPlotter(save_plots=save_plots, show_plots=show_plots)