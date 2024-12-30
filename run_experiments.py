import torch
import numpy as np
from pathlib import Path
import argparse
from typing import Dict

from experiments.utils import (
    ExperimentConfig,
    ExperimentTracker,
    run_single_trial
)
from data.mnist_sequence import get_mnist_dataloaders
from models.minimal_rnn import MinimalRNN
from models.vanilla_rnn import VanillaRNN
from core.mean_field import MeanFieldAnalyzer


def plot_results(results: Dict, save_path: Path, title: str):
    """Plot results with error bars."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    for param_value in results.keys():
        T_values = []
        means = []
        stds = []

        for T in results[param_value].keys():
            # Get accuracies for all trials
            accuracies = [trial['metrics']['final_test_accuracy']
                          for trial in results[param_value][T]]

            T_values.append(int(T))
            means.append(np.mean(accuracies))
            stds.append(np.std(accuracies))

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


def run_minimal_rnn_experiments(config: ExperimentConfig) -> Dict:
    """Run MinimalRNN experiments."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # mu_b_values = [-4, -2, 0, 2, 4, 6, 8]  # From paper
    mu_b_values = [-4]  # From paper
    tracker = ExperimentTracker(config, "minimal_rnn")

    for mu_b in mu_b_values:
        print(f"\nTesting MinimalRNN with μb = {mu_b}")

        # Get theoretical predictions
        analyzer = MeanFieldAnalyzer(mu_b=mu_b)
        theory = analyzer.verify_parameters(16.0)

        for T in config.T_values:
            print(f"  Sequence length T = {T}")

            # Get dataloaders
            train_loader, test_loader = get_mnist_dataloaders(
                T=T,
                batch_size=config.batch_size
            )

            # Run trials
            for trial in range(config.num_trials):
                print(f"    Trial {trial + 1}/{config.num_trials}")

                model = MinimalRNN(
                    input_size=784,
                    hidden_size=config.hidden_size,
                    mu_b=mu_b
                ).to(device)

                metrics = run_single_trial(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    config=config,
                    device=device
                )

                # Add theoretical predictions
                metrics['theory'] = {
                    'chi_1': float(theory['chi_1']),
                    'timescale': float(-1 / np.log(theory['chi_1']))
                }

                tracker.add_trial_result(mu_b, T, trial, metrics)

                print(f"      Test accuracy: {metrics['final_test_accuracy']:.3f}")

    return tracker.results


def run_vanilla_rnn_experiments(config: ExperimentConfig) -> Dict:
    """Run VanillaRNN experiments."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # sigma_w_values = [0.5, 0.8, 1.0, 1.2, 1.5]  # From paper
    sigma_w_values = [0.5]  # From paper

    tracker = ExperimentTracker(config, "vanilla_rnn")

    for sigma_w in sigma_w_values:
        print(f"\nTesting VanillaRNN with σw = {sigma_w}")

        for T in config.T_values:
            print(f"  Sequence length T = {T}")

            # Get dataloaders
            train_loader, test_loader = get_mnist_dataloaders(
                T=T,
                batch_size=config.batch_size
            )

            # Run trials
            for trial in range(config.num_trials):
                print(f"    Trial {trial + 1}/{config.num_trials}")

                model = VanillaRNN(
                    input_size=784,
                    hidden_size=config.hidden_size,
                    sigma_w=sigma_w
                ).to(device)

                metrics = run_single_trial(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    config=config,
                    device=device
                )

                tracker.add_trial_result(sigma_w, T, trial, metrics)

                print(f"      Test accuracy: {metrics['final_test_accuracy']:.3f}")

    return tracker.results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--num_trials', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--model', type=str, choices=['minimal', 'vanilla', 'both'],
                        default='both')
    parser.add_argument('--T_values', type=int, nargs='+', default=[10])
    args = parser.parse_args()

    config = ExperimentConfig(
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        num_trials=args.num_trials,
        save_dir=args.save_dir,
        T_values=args.T_values
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.model in ['minimal', 'both']:
        print("\nRunning MinimalRNN experiments...")
        minimal_results = run_minimal_rnn_experiments(config)
        plot_results(
            minimal_results,
            save_dir / 'minimal_rnn_results.png',
            "MinimalRNN Trainability"
        )

        # Save raw results
        torch.save(minimal_results, save_dir / 'minimal_rnn_results.pt')

    if args.model in ['vanilla', 'both']:
        print("\nRunning VanillaRNN experiments...")
        vanilla_results = run_vanilla_rnn_experiments(config)
        plot_results(
            vanilla_results,
            save_dir / 'vanilla_rnn_results.png',
            "VanillaRNN Trainability"
        )

        # Save raw results
        torch.save(vanilla_results, save_dir / 'vanilla_rnn_results.pt')

    print("\nExperiments complete!")


if __name__ == "__main__":
    main()