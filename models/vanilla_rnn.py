import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class VanillaRNN(nn.Module):
    """
    Implementation of Vanilla RNN for comparison with MinimalRNN.
    Uses same initialization scheme for fair comparison.

    Args:
        input_size: Dimension of input features
        hidden_size: Dimension of hidden state
        sigma_w: Standard deviation for hidden-to-hidden weights
        sigma_v: Standard deviation for input-to-hidden weights
        mu_b: Mean of bias term
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            sigma_w: float = 1.0,
            sigma_v: float = 0.025,  # From paper's experiments
            mu_b: float = 0.0,
            use_critical_init: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # RNN parameters
        self.W = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.V = nn.Parameter(torch.empty(hidden_size, input_size))
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
        Initialize using critical initialization from paper.
        For vanilla RNN, this means orthogonal initialization at edge of chaos.
        """
        # W initialization (orthogonal with scaling)
        W = torch.empty(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(W)
        W = W * self.sigma_w / np.sqrt(self.hidden_size)
        self.W.data.copy_(W)

        # V initialization
        nn.init.normal_(self.V, std=self.sigma_v / np.sqrt(self.input_size))

        # Bias initialization
        nn.init.constant_(self.b, self.mu_b)

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
        Forward pass of Vanilla RNN.

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

        outputs = []

        for t in range(seq_len):
            # Standard RNN update
            e = torch.mm(h, self.W.t()) + torch.mm(x[:, t], self.V.t()) + self.b
            h = torch.tanh(e)
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)

        return outputs, h

    def get_activation_statistics(self, h: torch.Tensor, x: torch.Tensor) -> dict:
        """
        Compute activation statistics for analysis.

        Args:
            h: Hidden state
            x: Input

        Returns:
            Dictionary containing activation statistics
        """
        e = torch.mm(h, self.W.t()) + torch.mm(x, self.V.t()) + self.b
        h_next = torch.tanh(e)

        return {
            'preact_mean': e.mean().item(),
            'preact_std': e.std().item(),
            'act_mean': h_next.mean().item(),
            'act_std': h_next.std().item()
        }


if __name__ == "__main__":
    # Basic test
    batch_size = 32
    seq_len = 10
    input_size = 100
    hidden_size = 128

    model = VanillaRNN(input_size, hidden_size)
    x = torch.randn(batch_size, seq_len, input_size)

    outputs, final_h = model(x)
    print(f"Output shape: {outputs.shape}")
    print(f"Final hidden state shape: {final_h.shape}")

    # Test activation statistics
    stats = model.get_activation_statistics(final_h, x[:, -1])
    print("\nActivation Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.3f}")