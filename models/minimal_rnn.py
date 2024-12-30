import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class MinimalRNN(nn.Module):
    """
    Implementation of MinimalRNN from the paper.

    Args:
        input_size: Dimension of input features
        hidden_size: Dimension of hidden state
        sigma_w: Standard deviation for hidden-to-hidden weights (default: 6.88 from paper)
        sigma_v: Standard deviation for input-to-hidden weights (default: 1.39 from paper)
        mu_b: Mean of bias term (default: 0.0)
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            sigma_w: float = 6.88,
            sigma_v: float = 1.39,
            mu_b: float = 0.0,
            use_critical_init: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input transformation (Φ function from paper)
        self.input_transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh()
        )

        # RNN parameters
        self.W = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.V = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.empty(hidden_size))

        # Initialize parameters
        self.sigma_w = sigma_w
        self.sigma_v = sigma_v
        self.mu_b = mu_b

        if use_critical_init:
            self._critical_initialization()
        else:
            self._default_initialization()

    def _critical_initialization(self):
        """
        Initialize parameters using critical initialization from paper.
        """
        # W initialization (hidden-to-hidden)
        nn.init.normal_(self.W, std=self.sigma_w / np.sqrt(self.hidden_size))

        # V initialization (input-to-hidden)
        nn.init.normal_(self.V, std=self.sigma_v / np.sqrt(self.hidden_size))

        # Bias initialization
        nn.init.constant_(self.b, self.mu_b)

        # Initialize input transform
        for name, param in self.input_transform.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _default_initialization(self):
        """
        Standard initialization for comparison.
        """
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.V)
        nn.init.zeros_(self.b)

    def forward(
            self,
            x: torch.Tensor,
            h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of MinimalRNN.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            h: Optional initial hidden state

        Returns:
            outputs: Sequence of hidden states
            h_n: Final hidden state
        """
        batch_size, seq_len, _ = x.size()

        if h is None:
            h = torch.zeros(batch_size, self.hidden_size,
                            device=x.device, dtype=x.dtype)

        # Prepare outputs tensor
        outputs = []

        # Process sequence
        for t in range(seq_len):
            # Transform input (Φ function)
            z = self.input_transform(x[:, t])

            # Update gate
            e = torch.mm(h, self.W.t()) + torch.mm(z, self.V.t()) + self.b
            u = torch.sigmoid(e)

            # State update
            h = u * h + (1 - u) * z
            outputs.append(h)

        # Stack outputs
        outputs = torch.stack(outputs, dim=1)

        return outputs, h

    def get_gate_statistics(self, h: torch.Tensor, x: torch.Tensor) -> dict:
        """
        Compute gate statistics for analysis.

        Args:
            h: Hidden state
            x: Input

        Returns:
            Dictionary containing gate statistics
        """
        z = self.input_transform(x)
        e = torch.mm(h, self.W.t()) + torch.mm(z, self.V.t()) + self.b
        u = torch.sigmoid(e)

        return {
            'u_mean': u.mean().item(),
            'u_std': u.std().item(),
            'e_mean': e.mean().item(),
            'e_std': e.std().item()
        }


if __name__ == "__main__":
    # Basic test of the model
    batch_size = 32
    seq_len = 10
    input_size = 100
    hidden_size = 128

    model = MinimalRNN(input_size, hidden_size)
    x = torch.randn(batch_size, seq_len, input_size)

    outputs, final_h = model(x)
    print(f"Output shape: {outputs.shape}")
    print(f"Final hidden state shape: {final_h.shape}")

    # Test gate statistics
    stats = model.get_gate_statistics(final_h, x[:, -1])
    print("\nGate Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.3f}")