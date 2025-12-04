"""Analyze fault-impact per layer for reliability-aware pruning."""
import torch
import torch.nn as nn
import copy
import sys
import os
from typing import Dict, List, Optional

# Import FaultInjector directly to avoid benchmarking.__init__ chain
import importlib.util
spec = importlib.util.spec_from_file_location("fault_injection",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmarking", "reliability", "fault_injection.py"))
fault_injection_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fault_injection_module)
FaultInjector = fault_injection_module.FaultInjector

from core.utils import cleanup_memory


class LayerFaultImpactAnalyzer:
    """Measures how critical each layer is under fault conditions."""

    def __init__(self, model: nn.Module, val_loader, device: torch.device,
                 ber_levels: List[float] = [1e-5, 1e-4, 5e-4]):
        """
        Initialize fault-impact analyzer.

        Args:
            model: Base model to analyze
            val_loader: Validation data loader
            device: Device to run on
            ber_levels: BER levels to test (focus on medium BER where impact matters)
        """
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.ber_levels = ber_levels
        self.fault_injector = FaultInjector()

    def compute_layer_fault_impacts(self, prunable_layers: List[str],
                                    repetitions: int = 10) -> Dict[str, float]:
        """
        For each layer, measure accuracy drop when faults are injected in that layer.

        Args:
            prunable_layers: List of layer names to analyze
            repetitions: Number of repetitions per BER level for statistical accuracy

        Returns:
            Dictionary mapping layer_name to average fault-impact score
            Higher score = more critical layer (faults cause bigger accuracy drop)
        """
        print("Computing baseline accuracy...")
        baseline_acc = self._test_accuracy(self.model)
        print(f"Baseline accuracy: {baseline_acc:.2f}%\n")

        fault_impacts = {}
        total_params = sum(p.numel() for p in self.model.parameters())

        for layer_idx, layer_name in enumerate(prunable_layers):
            print(f"[{layer_idx+1}/{len(prunable_layers)}] Testing fault impact for {layer_name}...")

            ber_impacts = []

            for ber in self.ber_levels:
                num_faults = max(1, int(ber * total_params))

                rep_impacts = []
                for rep in range(repetitions):
                    # Create faulty model with faults ONLY in this specific layer
                    faulty_model = copy.deepcopy(self.model)
                    self.fault_injector.inject_faults_inplace(
                        faulty_model, num_faults, [layer_name]
                    )

                    # Measure accuracy drop
                    faulty_acc = self._test_accuracy(faulty_model)
                    impact = baseline_acc - faulty_acc  # Positive = accuracy dropped
                    rep_impacts.append(max(0.0, impact))  # Clamp to 0 (no negative impacts)

                    del faulty_model
                    cleanup_memory()

                avg_impact_at_ber = sum(rep_impacts) / len(rep_impacts)
                ber_impacts.append(avg_impact_at_ber)
                print(f"  BER {ber:.0e}: {avg_impact_at_ber:.2f}% avg impact")

            # Average impact across all BER levels
            overall_impact = sum(ber_impacts) / len(ber_impacts)
            fault_impacts[layer_name] = overall_impact
            print(f"  → Overall fault impact: {overall_impact:.2f}%\n")

        return fault_impacts

    def _test_accuracy(self, model: nn.Module) -> float:
        """Quick accuracy test (subset of validation data for speed)."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                if batch_idx >= 15:  # Quick test with 15 batches
                    break
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        return (correct / total) * 100.0 if total > 0 else 0.0

    def save_fault_impacts(self, fault_impacts: Dict[str, float], filepath: str):
        """Save fault impacts to CSV for reuse."""
        import pandas as pd

        df = pd.DataFrame(list(fault_impacts.items()),
                         columns=['layer_name', 'fault_impact'])
        df = df.sort_values('fault_impact', ascending=False)  # Sort by impact
        df.to_csv(filepath, index=False)

        print(f"\n{'='*70}")
        print(f"Saved fault impacts to: {filepath}")
        print(f"{'='*70}")

    @staticmethod
    def load_fault_impacts(filepath: str) -> Dict[str, float]:
        """Load pre-computed fault impacts from CSV."""
        import pandas as pd

        df = pd.read_csv(filepath)
        fault_impacts = dict(zip(df['layer_name'], df['fault_impact']))

        print(f"Loaded {len(fault_impacts)} layer fault-impact scores from {filepath}")
        return fault_impacts

    def print_fault_impact_summary(self, fault_impacts: Dict[str, float]):
        """Print sorted summary of fault impacts."""
        print(f"\n{'='*70}")
        print("FAULT IMPACT SUMMARY (Higher = More Critical)")
        print(f"{'='*70}")
        print(f"{'Layer Name':<35} {'Fault Impact':>15} {'Classification':>15}")
        print(f"{'-'*70}")

        # Calculate thresholds
        impacts = list(fault_impacts.values())
        high_threshold = sorted(impacts, reverse=True)[len(impacts)//3] if len(impacts) >= 3 else max(impacts)
        medium_threshold = sorted(impacts, reverse=True)[2*len(impacts)//3] if len(impacts) >= 3 else max(impacts)/2

        sorted_impacts = sorted(fault_impacts.items(), key=lambda x: x[1], reverse=True)

        for layer, impact in sorted_impacts:
            if impact >= high_threshold:
                classification = "CRITICAL"
            elif impact >= medium_threshold:
                classification = "MEDIUM"
            else:
                classification = "LOW"

            print(f"{layer:<35} {impact:>14.2f}% {classification:>15}")

        print(f"{'='*70}\n")
