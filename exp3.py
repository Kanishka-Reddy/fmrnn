import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from torchvision import datasets, transforms
from tqdm import tqdm
from dataclasses import dataclass, field
import math

from models.minimal_rnn import MinimalRNN
from models.vanilla_rnn import VanillaRNN
from data.mnist_sequence import get_mnist_dataloaders

# Import your VanillaRNN, MinimalRNN, and Dataset classes
from models import MNISTSequenceDataset

@dataclass
class ExperimentConfig:
    """Configuration for Figure 3 experiments."""
    hidden_size: int = 128
    batch_size: int = 32
    epochs: int = 5
    num_trials: int = 3
    sigma_x: float = 1.0
    T_values: List[int] = field(default_factory=lambda: [196, 784])
    learning_rate: float = 0.001

@dataclass
class InitializationConfig:
    """Defines critical and off-critical initialization parameters."""
    critical_sigma_w: float
    critical_sigma_v: float
    critical_mu_b: float = 0.0  # Optional, primarily for MinimalRNN
    off_critical_factor: float = 1.1  # Factor to perturb critical values

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, save_dir: Path):
        self.config = config
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def _get_model(self, model_type: str, init_config: InitializationConfig, critical: bool) -> nn.Module:
        """Initialize the model with critical or off-critical parameters."""
        sigma_w = (init_config.critical_sigma_w
                   if critical else init_config.critical_sigma_w * init_config.off_critical_factor)
        sigma_v = (init_config.critical_sigma_v
                   if critical else init_config.critical_sigma_v * init_config.off_critical_factor)
        mu_b = init_config.critical_mu_b

        model_params = {
            'input_size': 784,
            'hidden_size': self.config.hidden_size,
            'sigma_w': sigma_w,
            'sigma_v': sigma_v,
            'mu_b': mu_b
        }

        if model_type == 'vanilla':
            return VanillaRNN(**model_params).to(self.device)
        elif model_type == 'minimal':
            return MinimalRNN(**model_params).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _run_single_trial(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            test_loader: DataLoader
    ) -> Dict:
        """Train and evaluate a single model for one trial."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        epoch_accuracies = []

        for epoch in range(self.config.epochs):
            # Training phase
            model.train()
            for sequences, targets in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs, _ = model(sequences)
                loss = criterion(outputs[:, -1], targets)
                loss.backward()
                optimizer.step()

            # Evaluation phase
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for sequences, targets in test_loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)

                    outputs, _ = model(sequences)
                    _, predicted = outputs[:, -1].max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            accuracy = correct / total
            epoch_accuracies.append(accuracy)

            print(f"Epoch {epoch + 1}/{self.config.epochs}, Accuracy: {accuracy:.3f}")

        return {'epoch_accuracies': epoch_accuracies}

    def run_experiment(
            self,
            model_type: str,
            init_config: InitializationConfig,
            critical: bool
    ) -> Dict:
        """
        Run an experiment for a given model type and initialization configuration.

        Args:
            model_type: 'vanilla' or 'minimal'.
            init_config: Initialization configuration (critical/off-critical values).
            critical: If True, use critical initialization. If False, use off-critical.

        Returns:
            A dictionary of results with epoch-wise accuracies.
        """
        results = {}

        for T in self.config.T_values:
            print(f"\nRunning {model_type.upper()} model with T={T}, "
                  f"{'Critical' if critical else 'Off-Critical'} Initialization")

            # Prepare data loaders
            train_loader, test_loader = get_mnist_sequence_loaders(
                T=T,
                batch_size=self.config.batch_size,
                sigma_x=self.config.sigma_x
            )

            trial_accuracies = []
            for trial in range(self.config.num_trials):
                print(f"  Trial {trial + 1}/{self.config.num_trials}")

                # Initialize model
                model = self._get_model(model_type, init_config, critical)

                # Train and evaluate
                trial_result = self._run_single_trial(model, train_loader, test_loader)
                trial_accuracies.append(trial_result['epoch_accuracies'])

            # Store results as the average over trials
            results[str(T)] = np.mean(trial_accuracies, axis=0).tolist()

        return results

    def _run_single_trial_with_steps(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            test_loader: DataLoader
    ) -> Dict:
        """
        Train and evaluate a single model, tracking test accuracy at logarithmic optimization steps.

        Returns:
            A dictionary mapping optimization steps to test accuracy.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        # Log-spaced steps to track accuracy
        log_steps = [int(10 ** i) for i in range(1, int(math.log10(5e4)) + 1)]
        step_counter = 0
        results = {}

        for epoch in range(self.config.epochs):
            model.train()
            for sequences, targets in train_loader:
                step_counter += 1
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs, _ = model(sequences)
                loss = criterion(outputs[:, -1], targets)
                loss.backward()
                optimizer.step()

                # Evaluate at log-spaced steps
                if step_counter in log_steps:
                    accuracy = self._evaluate_model(model, test_loader)
                    results[step_counter] = accuracy
                    print(f"Step {step_counter}, Accuracy: {accuracy:.3f}")

                # Early stop if max steps reached
                if step_counter >= 5e4:  # Matches plot's x-axis limit
                    break
            if step_counter >= 5e4:
                break

        return results

    def _evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> float:
        """Evaluate model on the test set and return accuracy."""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                outputs, _ = model(sequences)
                _, predicted = outputs[:, -1].max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return correct / total

    def run_experiment_with_variants(
            self,
            model_type: str,
            init_config: InitializationConfig
    ) -> Dict:
        """
        Run an experiment for a given model type with multiple initialization conditions.

        Returns:
            A dictionary containing results for critical/off-critical and orthogonal/gaussian setups.
        """
        results = {  # Organize results by condition
            "orthogonal_critical": {},
            "orthogonal_off_critical": {},
            "gaussian_critical": {},
            "gaussian_off_critical": {}
        }

        for T in self.config.T_values:
            print(f"\nRunning {model_type.upper()} with T={T}")

            # Prepare data loaders
            train_loader, test_loader = get_mnist_sequence_loaders(
                T=T,
                batch_size=self.config.batch_size,
                sigma_x=self.config.sigma_x
            )

            for condition in results.keys():
                print(f"  Condition: {condition.replace('_', ', ').capitalize()}")

                # Initialize model with specific initialization
                critical = "critical" in condition
                orthogonal = "orthogonal" in condition

                model = self._get_model_variant(
                    model_type, init_config, critical, orthogonal
                )

                # Train and track accuracy
                results[condition][T] = self._run_single_trial_with_steps(
                    model, train_loader, test_loader
                )

        return results

    def _get_model_variant(
            self,
            model_type: str,
            init_config: InitializationConfig,
            critical: bool,
            orthogonal: bool
    ) -> nn.Module:
        """Initialize model with specified initialization type."""
        sigma_w = (init_config.critical_sigma_w
                   if critical else init_config.critical_sigma_w * init_config.off_critical_factor)
        sigma_v = (init_config.critical_sigma_v
                   if critical else init_config.critical_sigma_v * init_config.off_critical_factor)
        mu_b = init_config.critical_mu_b

        model_params = {
            'input_size': 784,
            'hidden_size': self.config.hidden_size,
            'sigma_w': sigma_w,
            'sigma_v': sigma_v,
            'mu_b': mu_b
        }

        # Select appropriate model class
        model_class = VanillaRNN if model_type == 'vanilla' else MinimalRNN
        model = model_class(**model_params)

        # Initialize weights
        with torch.no_grad():
            if orthogonal:
                nn.init.orthogonal_(model.W)
                nn.init.orthogonal_(model.V)
            else:
                nn.init.normal_(model.W, std=sigma_w / math.sqrt(self.config.hidden_size))
                nn.init.normal_(model.V, std=sigma_v / math.sqrt(self.config.hidden_size))

        return model.to(self.device)

import matplotlib.pyplot as plt
import numpy as np

class LearningDynamicsPlotter:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_dynamics(
        self,
        results: Dict,
        model_type: str,
        T_values: List[int],
        save_name: str = "learning_dynamics.png"
    ):
        """
        Generate learning dynamics plots for VanillaRNN and MinimalRNN.

        Args:
            results: Dictionary containing results for all initialization variants.
            model_type: "vanilla" or "minimal".
            T_values: List of sequence lengths (e.g., [196, 784]).
            save_name: Filename for saving the plot.
        """
        fig, axes = plt.subplots(1, len(T_values), figsize=(15, 6), sharey=True)
        colors = {
            "orthogonal_critical": "blue",
            "gaussian_critical": "red",
            "orthogonal_off_critical": "green",
            "gaussian_off_critical": "black"
        }
        linestyles = {
            "critical": "-",
            "off_critical": "--"
        }

        for i, T in enumerate(T_values):
            ax = axes[i]
            ax.set_title(f"{model_type.capitalize()} RNN, T={T}")
            ax.set_xlabel("Optimization Steps (t)")
            ax.set_xscale("log")
            if i == 0:
                ax.set_ylabel("Accuracy")

            for condition, color in colors.items():
                if T not in results[condition]:
                    continue
                steps = sorted(results[condition][T].keys())
                accuracies = [results[condition][T][step] for step in steps]

                linestyle = linestyles["critical"] if "critical" in condition else linestyles["off_critical"]
                label = condition.replace("_", ", ").capitalize()
                ax.plot(steps, accuracies, label=label, color=color, linestyle=linestyle)

            ax.legend(loc="lower right")

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Learning dynamics plot saved to {save_path}")


