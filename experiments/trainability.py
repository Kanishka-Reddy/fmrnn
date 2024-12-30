import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

from models.minimal_rnn import MinimalRNN
from models.vanilla_rnn import VanillaRNN
from data.mnist_sequence import get_mnist_dataloaders, Trainer
from core.mean_field import MeanFieldAnalyzer


class TrainabilityExperiment:
    """
    Reproduces Figure 2 from paper: Relationship between theory and trainability.

    Args:
        hidden_size: Dimension of hidden state
        T_values: List of sequence lengths to test
        max_epochs: Maximum training epochs
        device: Device to train on
        results_dir: Directory to save results
    """

    def __init__(
            self,
            hidden_size: int = 128,
            T_values: List[int] = [10, 100, 1000],
            max_epochs: int = 10,
            device: Optional[str] = None,
            results_dir: str = './results'
    ):
        self.hidden_size = hidden_size
        self.T_values = T_values
        self.max_epochs = max_epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_minimal_rnn_experiment(
            self,
            mu_b_values: List[float] = [-4, -2, 0, 2, 4, 6, 8]
    ) -> Dict:
        """Run trainability experiment for MinimalRNN."""
        results = {}

        for mu_b in mu_b_values:
            results[mu_b] = {}

            # Compute theoretical timescale
            analyzer = MeanFieldAnalyzer(mu_b=mu_b)
            theory_result = analyzer.verify_parameters(16.0)  # q* from our analysis

            for T in self.T_values:
                print(f"\nTesting μb={mu_b}, T={T}")

                # Create model and optimizer
                model = MinimalRNN(
                    input_size=784,
                    hidden_size=self.hidden_size,
                    mu_b=mu_b
                )

                optimizer = optim.Adam(model.parameters())
                criterion = nn.CrossEntropyLoss()

                # Get data
                train_loader, test_loader = get_mnist_dataloaders(
                    T=T,
                    batch_size=128
                )

                # Train
                trainer = Trainer(
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=self.device,
                    clip_grad=1.0
                )

                epoch_results = []
                for epoch in range(self.max_epochs):
                    train_metrics = trainer.train_epoch(train_loader)
                    test_metrics = trainer.evaluate(test_loader)

                    epoch_results.append({
                        'epoch': epoch,
                        'train': train_metrics,
                        'test': test_metrics
                    })

                    print(f"Epoch {epoch}: "
                          f"Train acc={train_metrics['accuracy']:.3f}, "
                          f"Test acc={test_metrics['accuracy']:.3f}")

                results[mu_b][T] = {
                    'training': epoch_results,
                    'theory': {
                        'chi_1': theory_result['chi_1'],
                        'timescale': -1 / np.log(theory_result['chi_1'])
                    }
                }

        # Save results
        with open(self.results_dir / 'minimal_rnn_results.json', 'w') as f:
            json.dump(results, f)

        return results

    def run_vanilla_rnn_experiment(
            self,
            sigma_w_values: List[float] = [0.5, 0.8, 1.0, 1.2, 1.5]
    ) -> Dict:
        """Run trainability experiment for VanillaRNN."""
        results = {}

        for sigma_w in sigma_w_values:
            results[sigma_w] = {}

            for T in self.T_values:
                print(f"\nTesting σw={sigma_w}, T={T}")

                # Create model and optimizer
                model = VanillaRNN(
                    input_size=784,
                    hidden_size=self.hidden_size,
                    sigma_w=sigma_w
                )

                optimizer = optim.Adam(model.parameters())
                criterion = nn.CrossEntropyLoss()

                # Get data
                train_loader, test_loader = get_mnist_dataloaders(
                    T=T,
                    batch_size=128
                )

                # Train
                trainer = Trainer(
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=self.device,
                    clip_grad=1.0
                )

                epoch_results = []
                for epoch in range(self.max_epochs):
                    train_metrics = trainer.train_epoch(train_loader)
                    test_metrics = trainer.evaluate(test_loader)

                    epoch_results.append({
                        'epoch': epoch,
                        'train': train_metrics,
                        'test': test_metrics
                    })

                    print(f"Epoch {epoch}: "
                          f"Train acc={train_metrics['accuracy']:.3f}, "
                          f"Test acc={test_metrics['accuracy']:.3f}")

                results[sigma_w][T] = {
                    'training': epoch_results
                }

        # Save results
        with open(self.results_dir / 'vanilla_rnn_results.json', 'w') as f:
            json.dump(results, f)

        return results

    def plot_results(self, minimal_results: Dict, vanilla_results: Dict):
        """Reproduce Figure 2 from paper."""
        plt.figure(figsize=(15, 5))

        # Plot MinimalRNN results
        plt.subplot(121)
        self._plot_model_results(minimal_results, 'MinimalRNN', x_label='μb')

        # Plot VanillaRNN results
        plt.subplot(122)
        self._plot_model_results(vanilla_results, 'VanillaRNN', x_label='σw')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'trainability_results.png')
        plt.close()

    def _plot_model_results(self, results: Dict, model_name: str, x_label: str):
        """Helper for plotting results."""
        for param_value in results.keys():
            accuracies = []
            for T in self.T_values:
                # Get final test accuracy
                final_acc = results[param_value][T]['training'][-1]['test']['accuracy']
                accuracies.append(final_acc)

            plt.plot(self.T_values, accuracies, 'o-', label=f'{x_label}={param_value}')

        plt.xlabel('Sequence Length (T)')
        plt.ylabel('Test Accuracy')
        plt.title(f'{model_name} Trainability')
        plt.legend()
        plt.xscale('log')
        plt.grid(True)