import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import time
from torch.utils.data import DataLoader


@dataclass
class ExperimentConfig:
    """Configuration for trainability experiments."""
    hidden_size: int = 128
    batch_size: int = 128
    max_epochs: int = 10
    num_trials: int = 5
    learning_rate: float = 0.001
    clip_grad: float = 1.0
    T_values: List[int] = None
    save_dir: str = './results'

    def __post_init__(self):
        if self.T_values is None:
            self.T_values = [10] # [10, 100, 1000]
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)


class ExperimentTracker:
    """Tracks experiment progress and saves results."""

    def __init__(self, config: ExperimentConfig, experiment_name: str):
        self.config = config
        self.experiment_name = experiment_name
        self.results = {}
        self.start_time = time.time()

    def add_trial_result(
            self,
            param_value: float,
            T: int,
            trial: int,
            metrics: Dict
    ):
        """Add results from a single trial."""
        if param_value not in self.results:
            self.results[param_value] = {}
        if T not in self.results[param_value]:
            self.results[param_value][T] = []

        self.results[param_value][T].append({
            'trial': trial,
            'metrics': metrics
        })

    def save_results(self):
        """Save results to disk."""
        save_path = Path(self.config.save_dir) / f'{self.experiment_name}_results.json'

        # Compute summary statistics
        summary = self._compute_summary_statistics()

        # Save both raw results and summary
        results_dict = {
            'config': vars(self.config),
            'raw_results': self.results,
            'summary': summary,
            'runtime': time.time() - self.start_time
        }

        with open(save_path, 'w') as f:
            json.dump(results_dict, f)

    def _compute_summary_statistics(self) -> Dict:
        """Compute summary statistics across trials."""
        summary = {}

        for param_value in self.results:
            summary[param_value] = {}
            for T in self.results[param_value]:
                # Get final accuracies across trials
                final_accuracies = [
                    trial['metrics']['final_test_accuracy']
                    for trial in self.results[param_value][T]
                ]

                summary[param_value][T] = {
                    'mean_accuracy': np.mean(final_accuracies),
                    'std_accuracy': np.std(final_accuracies),
                    'min_accuracy': np.min(final_accuracies),
                    'max_accuracy': np.max(final_accuracies)
                }

        return summary


def run_single_trial(
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: ExperimentConfig,
        device: torch.device
) -> Dict:
    """Run a single training trial."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Training history
    train_accuracies = []
    test_accuracies = []

    for epoch in range(config.max_epochs):
        # Train
        model.train()
        train_correct = 0
        train_total = 0

        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(sequences)
            logits = outputs[:, -1]  # Use final timestep

            loss = criterion(logits, labels)
            loss.backward()

            if config.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.clip_grad
                )

            optimizer.step()

            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total
        train_accuracies.append(train_acc)

        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)

                outputs, _ = model(sequences)
                logits = outputs[:, -1]

                preds = logits.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        test_acc = test_correct / test_total
        test_accuracies.append(test_acc)

    return {
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'final_train_accuracy': train_accuracies[-1],
        'final_test_accuracy': test_accuracies[-1]
    }


def analyze_convergence(accuracies: List[float], threshold: float = 0.9) -> Dict:
    """Analyze convergence behavior."""
    converged = max(accuracies) >= threshold
    if converged:
        convergence_epoch = next(
            i for i, acc in enumerate(accuracies)
            if acc >= threshold
        )
    else:
        convergence_epoch = None

    return {
        'converged': converged,
        'convergence_epoch': convergence_epoch,
        'max_accuracy': max(accuracies),
        'final_accuracy': accuracies[-1]
    }


def save_experiment_plot(
        results_dict: Dict,
        save_path: Path,
        title: str = "Trainability Results"
):
    """Create and save plot of experiment results."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    summary = results_dict['summary']
    config = results_dict['config']
    T_values = config['T_values']

    for param_value in summary:
        means = [summary[param_value][str(T)]['mean_accuracy'] for T in T_values]
        stds = [summary[param_value][str(T)]['std_accuracy'] for T in T_values]

        plt.errorbar(
            T_values,
            means,
            yerr=stds,
            label=f'param={param_value}',
            marker='o'
        )

    plt.xscale('log')
    plt.xlabel('Sequence Length (T)')
    plt.ylabel('Test Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()