import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from models.minimal_rnn import MinimalRNN
from models.vanilla_rnn import VanillaRNN

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
matplotlib.use('TkAgg')  # Use TkAgg backend, or try 'Agg', 'Qt5Agg', or 'MacOSX' based on your system.


class MNISTSequenceDataset(Dataset):
    """
    Custom Dataset for MNIST-derived sequence task.
    Args:
        mnist_data: Torchvision MNIST dataset.
        seq_len: Length of the sequence (T).
    """

    def __init__(self, mnist_data, seq_len=196):
        self.mnist_data = mnist_data
        self.seq_len = seq_len
        self.input_dim = 784 // seq_len  # Pixels per timestep

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        img, label = self.mnist_data[idx]
        img = img.view(-1)  # Flatten the image (28x28 -> 784)

        # First timestep contains actual MNIST data (split into seq_len parts)
        first_timestep = img[:self.input_dim * self.seq_len].view(self.seq_len, -1)

        # Remaining timesteps are random noise
        noise = torch.randn((self.seq_len - 1, self.input_dim))

        # Combine first timestep (data) with noise
        sequence = torch.cat([first_timestep, noise], dim=0)

        return sequence, label


def get_mnist_sequence_dataloader(seq_len, batch_size=32):
    """
    Create DataLoader for MNIST sequence task.
    Args:
        seq_len: Sequence length (T).
        batch_size: Batch size for DataLoader.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_dataset = MNISTSequenceDataset(mnist_train, seq_len=seq_len)
    test_dataset = MNISTSequenceDataset(mnist_test, seq_len=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_rnn(model, train_loader, test_loader, num_epochs, device):
    """
    Training loop for RNN models.

    Args:
        model: The RNN model (VanillaRNN or MinimalRNN).
        train_loader: DataLoader for training data.
        test_loader: DataLoader for test data.
        num_epochs: Number of training epochs.
        device: Device to train on (CPU or GPU).

    Returns:
        train_acc_history: Training accuracy over epochs.
        test_acc_history: Test accuracy over epochs.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_acc_history = []
    test_acc_history = []

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0

        for sequences, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            sequences, labels = sequences.to(device), labels.to(device)

            # Forward pass
            outputs, _ = model(sequences)
            logits = outputs[:, -1, :]  # Use the last timestep for classification

            # Compute loss
            loss = criterion(logits, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute training accuracy
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_acc_history.append(train_acc)

        # Evaluate on test set
        test_acc = evaluate_rnn(model, test_loader, device)
        test_acc_history.append(test_acc)

        print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

    return train_acc_history, test_acc_history


def evaluate_rnn(model, test_loader, device):
    """
    Evaluate RNN model on test data.

    Args:
        model: The trained RNN model.
        test_loader: DataLoader for test data.
        device: Device to evaluate on.

    Returns:
        Test accuracy.
    """
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            outputs, _ = model(sequences)
            logits = outputs[:, -1, :]  # Use the last timestep for classification

            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total


def plot_learning_curves(train_acc_history, test_acc_history, title):
    """
    Plot learning curves for training and testing accuracy.

    Args:
        train_acc_history: List of training accuracies.
        test_acc_history: List of test accuracies.
        title: Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(test_acc_history, label='Test Accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    seq_len = 196  # Change to 784 for longer sequences
    batch_size = 32
    num_epochs = 2
    hidden_size = 64
    input_size = 784 // seq_len  # Pixels per timestep
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, test_loader = get_mnist_sequence_dataloader(seq_len, batch_size)

    # Define models
    models = {
        "VanillaRNN": VanillaRNN,
        "MinimalRNN": MinimalRNN,
    }

    init_conditions = [
        {"name": "Critical-Orthogonal", "use_critical_init": True},
        {"name": "Critical-Gaussian", "use_critical_init": True, "sigma_w": 1.0, "sigma_v": 1.0},
        {"name": "Off-Critical-Orthogonal", "use_critical_init": False},
        {"name": "Off-Critical-Gaussian", "use_critical_init": False, "sigma_w": 1.0, "sigma_v": 1.0},
    ]

    for model_name, model_class in models.items():
        for condition in init_conditions:
            print(f"Running {model_name} under {condition['name']}...")
            # Pass parameters to the constructor
            model = model_class(
                input_size=input_size,
                hidden_size=hidden_size,
                sigma_w=condition.get("sigma_w", 1.0),
                sigma_v=condition.get("sigma_v", 0.025),
                use_critical_init=condition["use_critical_init"]
            )
            train_acc, test_acc = train_rnn(model, train_loader, test_loader, num_epochs, device)
            plot_learning_curves(train_acc, test_acc, f"{model_name} - {condition['name']} (T={seq_len})")

