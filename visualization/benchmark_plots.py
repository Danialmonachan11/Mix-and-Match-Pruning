"""Visualization utilities for benchmarking results."""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Optional


class BenchmarkPlotter:
    """Plotting utilities for benchmark results."""
    
    def __init__(self, save_dir: str = "benchmark_plots"):
        """
        Initialize benchmark plotter.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set matplotlib backend for headless environments
        plt.switch_backend('Agg')
        
        # Set style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # Set color palette
        self.method_colors = {
            'baseline': 'black',
            'magnitude_global': 'blue',
            'magnitude_layerwise': 'lightblue',
            'random_global': 'red',
            'random_layerwise': 'lightcoral',
            'wanda': 'green',
            'wanda_magnitude': 'lightgreen',
            'snip': 'orange',
            'grasp': 'purple',
            'ga_nsga2': 'gold'
        }
    
    def plot_accuracy_comparison(self, benchmark_results: Dict[str, Any], filename: str = None):
        """
        Plot accuracy comparison across methods.
        
        Args:
            benchmark_results: Dictionary of benchmark results
            filename: Output filename
        """
        methods = []
        accuracies = []
        method_types = []
        
        for method_name, results in benchmark_results.items():
            if results.get('success', True):
                methods.append(method_name)
                accuracies.append(results['accuracy'])
                method_types.append(results.get('method_type', 'unknown'))
        
        if not methods:
            print("No valid results for accuracy comparison")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create bar plot
        bars = plt.bar(range(len(methods)), accuracies, 
                      color=[self.method_colors.get(m, 'gray') for m in methods],
                      alpha=0.7)
        
        plt.xlabel('Pruning Method')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Comparison Across Pruning Methods')
        plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{acc:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"accuracy_comparison_{timestamp}.png"
        
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Accuracy comparison plot saved to {filepath}")
    
    def plot_reliability_heatmap(self, benchmark_results: Dict[str, Any], 
                                fault_levels: List[int], filename: str = None):
        """
        Plot reliability heatmap showing performance degradation under faults.
        
        Args:
            benchmark_results: Dictionary of benchmark results
            fault_levels: List of fault levels tested
            filename: Output filename
        """
        # Prepare data for heatmap
        methods = []
        reliability_data = []
        
        for method_name, results in benchmark_results.items():
            if (results.get('success', True) and 
                results.get('reliability_results') and
                'fault_levels' in results['reliability_results']):
                
                methods.append(method_name)
                method_reliabilities = []
                
                for fault_level in fault_levels:
                    if fault_level in results['reliability_results']['fault_levels']:
                        rel_stats = results['reliability_results']['fault_levels'][fault_level]
                        method_reliabilities.append(rel_stats['mean'])
                    else:
                        method_reliabilities.append(0.0)
                
                reliability_data.append(method_reliabilities)
        
        if not methods:
            print("No valid reliability results for heatmap")
            return
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        reliability_array = np.array(reliability_data)
        
        # Create heatmap
        sns.heatmap(reliability_array, 
                   xticklabels=[f'{fl} faults' for fl in fault_levels],
                   yticklabels=methods,
                   annot=True, fmt='.2f', cmap='RdYlGn',
                   cbar_kws={'label': 'Accuracy under Faults (%)'})
        
        plt.title('Reliability Heatmap: Performance Under Hardware Faults')
        plt.xlabel('Fault Level')
        plt.ylabel('Pruning Method')
        plt.tight_layout()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reliability_heatmap_{timestamp}.png"
        
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Reliability heatmap saved to {filepath}")
    
    def plot_pareto_comparison(self, benchmark_results: Dict[str, Any], filename: str = None):
        """
        Plot Pareto comparison of accuracy vs sparsity.
        
        Args:
            benchmark_results: Dictionary of benchmark results
            filename: Output filename
        """
        plt.figure(figsize=(12, 8))
        
        for method_name, results in benchmark_results.items():
            if results.get('success', True):
                accuracy = results['accuracy']
                sparsity = results['sparsity']
                method_type = results.get('method_type', 'unknown')
                
                color = self.method_colors.get(method_name, 'gray')
                marker = 'o' if method_type == 'baseline' else 's' if method_type == 'classical' else '^' if method_type == 'sota' else 'D'
                
                plt.scatter(sparsity, accuracy, c=color, s=100, alpha=0.7, 
                          marker=marker, label=method_name, edgecolors='black')
        
        plt.xlabel('Sparsity (%)')
        plt.ylabel('Accuracy (%)')
        plt.title('Pareto Comparison: Accuracy vs Sparsity')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pareto_comparison_{timestamp}.png"
        
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Pareto comparison plot saved to {filepath}")
    
    def plot_reliability_degradation_curves(self, benchmark_results: Dict[str, Any],
                                          fault_levels: List[int], filename: str = None):
        """
        Plot reliability degradation curves for each method.
        
        Args:
            benchmark_results: Dictionary of benchmark results
            fault_levels: List of fault levels tested
            filename: Output filename
        """
        plt.figure(figsize=(12, 8))
        
        for method_name, results in benchmark_results.items():
            if (results.get('success', True) and 
                results.get('reliability_results') and
                'fault_levels' in results['reliability_results']):
                
                reliabilities = []
                available_fault_levels = []
                
                for fault_level in fault_levels:
                    if fault_level in results['reliability_results']['fault_levels']:
                        rel_stats = results['reliability_results']['fault_levels'][fault_level]
                        reliabilities.append(rel_stats['mean'])
                        available_fault_levels.append(fault_level)
                
                if reliabilities:
                    color = self.method_colors.get(method_name, 'gray')
                    plt.plot(available_fault_levels, reliabilities, 
                           marker='o', linewidth=2, label=method_name, color=color)
        
        plt.xlabel('Number of Injected Faults')
        plt.ylabel('Accuracy under Faults (%)')
        plt.title('Reliability Degradation Curves')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xscale('log')
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reliability_curves_{timestamp}.png"
        
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Reliability curves plot saved to {filepath}")
    
    def plot_method_type_comparison(self, benchmark_results: Dict[str, Any], filename: str = None):
        """
        Plot comparison grouped by method type.
        
        Args:
            benchmark_results: Dictionary of benchmark results
            filename: Output filename
        """
        # Group methods by type
        method_groups = {}
        for method_name, results in benchmark_results.items():
            if results.get('success', True):
                method_type = results.get('method_type', 'unknown')
                if method_type not in method_groups:
                    method_groups[method_type] = []
                method_groups[method_type].append((method_name, results))
        
        if not method_groups:
            print("No valid results for method type comparison")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot accuracy by method type
        for method_type, methods in method_groups.items():
            accuracies = [results['accuracy'] for _, results in methods]
            method_names = [name for name, _ in methods]
            
            x_pos = np.arange(len(method_names))
            ax1.bar(x_pos, accuracies, alpha=0.7, label=method_type)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(method_names, rotation=45, ha='right')
        
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy by Method Type')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot average reliability by method type
        type_avg_reliability = {}
        for method_type, methods in method_groups.items():
            reliabilities = []
            for _, results in methods:
                if results.get('reliability_results'):
                    fault_reliabilities = [
                        stats['mean'] for stats in 
                        results['reliability_results']['fault_levels'].values()
                    ]
                    if fault_reliabilities:
                        reliabilities.append(np.mean(fault_reliabilities))
            
            if reliabilities:
                type_avg_reliability[method_type] = np.mean(reliabilities)
        
        if type_avg_reliability:
            types = list(type_avg_reliability.keys())
            avg_rels = list(type_avg_reliability.values())
            
            ax2.bar(types, avg_rels, alpha=0.7, color=['red', 'blue', 'green', 'orange'][:len(types)])
            ax2.set_ylabel('Average Reliability (%)')
            ax2.set_title('Average Reliability by Method Type')
            ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"method_type_comparison_{timestamp}.png"
        
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Method type comparison plot saved to {filepath}")
    
    def create_comprehensive_report(self, benchmark_results: Dict[str, Any], 
                                  fault_levels: List[int], filename: str = None):
        """
        Create comprehensive benchmark report with all plots.
        
        Args:
            benchmark_results: Dictionary of benchmark results
            fault_levels: List of fault levels tested
            filename: Output filename prefix
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_report_{timestamp}"
        
        print(f"Creating comprehensive benchmark report with prefix: {filename}")
        
        # Plot accuracy comparison
        self.plot_accuracy_comparison(benchmark_results, f"{filename}_accuracy.png")
        
        # Plot reliability heatmap
        self.plot_reliability_heatmap(benchmark_results, fault_levels, f"{filename}_reliability_heatmap.png")
        
        # Plot Pareto comparison
        self.plot_pareto_comparison(benchmark_results, f"{filename}_pareto.png")
        
        # Plot reliability curves
        self.plot_reliability_degradation_curves(benchmark_results, fault_levels, f"{filename}_reliability_curves.png")
        
        # Plot method type comparison
        self.plot_method_type_comparison(benchmark_results, f"{filename}_method_types.png")
        
        print(f"Comprehensive benchmark report completed in directory: {self.save_dir}")


def create_benchmark_plotter(save_dir: str = "benchmark_plots") -> BenchmarkPlotter:
    """Factory function to create benchmark plotter."""
    return BenchmarkPlotter(save_dir)