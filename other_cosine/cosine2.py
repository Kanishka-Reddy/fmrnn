import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.stats import norm


class MinimalRNN:
    def __init__(self, hidden_dim, sigma_w=6.88, sigma_v=1.39, R=0.46, mu_b=0.0, tied=True):
        self.hidden_dim = hidden_dim
        self.sigma_w = sigma_w
        self.sigma_v = sigma_v
        self.R = R
        self.mu_b = mu_b
        self.tied = tied

        # Initialize weights
        scale_w = sigma_w / np.sqrt(hidden_dim)
        scale_v = sigma_v / np.sqrt(hidden_dim)

        if tied:
            self.W = np.random.randn(hidden_dim, hidden_dim) * scale_w
            self.V = np.random.randn(hidden_dim, hidden_dim) * scale_v
        else:
            # For untied weights, create a list of weights for each timestep
            self.W = []
            self.V = []

        self.b = np.full(hidden_dim, mu_b)

    def get_weights(self, t):
        """Get weights for timestep t"""
        if self.tied:
            return self.W, self.V

        # Generate new weights if needed for untied case
        while len(self.W) <= t:
            scale_w = self.sigma_w / np.sqrt(self.hidden_dim)
            scale_v = self.sigma_v / np.sqrt(self.hidden_dim)
            self.W.append(np.random.randn(self.hidden_dim, self.hidden_dim) * scale_w)
            self.V.append(np.random.randn(self.hidden_dim, self.hidden_dim) * scale_v)

        return self.W[t], self.V[t]

    def forward(self, x, h, t):
        W, V = self.get_weights(t)
        e = np.dot(W, h) + np.dot(V, x) + self.b
        u = 1 / (1 + np.exp(-e))
        h_new = u * h + (1 - u) * x
        return h_new


def theoretical_prediction(q_star, mu_b, sigma_w, t, Sigma_12=1.0):
    """
    Compute theoretical prediction for cosine similarity
    Based on equations from section 3.2 of the paper
    """

    def integrand(z1, z2, q_star, mu_b):
        u1 = 1 / (1 + np.exp(-(np.sqrt(q_star) * z1 + mu_b)))
        u2 = 1 / (1 + np.exp(-(np.sqrt(q_star) * z2 + mu_b)))
        return u1 * u2 * norm.pdf(z1) * norm.pdf(z2)

    # Compute χ_c* (equation 9 in paper)
    chi_c, _ = dblquad(lambda z1, z2: integrand(z1, z2, q_star, mu_b),
                       -np.inf, np.inf, -np.inf, np.inf)

    # Compute predicted cosine similarity
    if Sigma_12 == 1.0:
        c_t = chi_c ** t
    else:
        c_t = Sigma_12 * (1 - (1 - chi_c) ** t)

    return c_t


def run_simulation(n_steps=100, n_trials=1000, hidden_dim=512, mu_b=-3.0, tied=True):
    """Run simulation with either tied or untied weights"""
    similarities = np.zeros(n_steps)
    R = 0.46

    for trial in range(n_trials):
        rnn = MinimalRNN(hidden_dim, mu_b=mu_b, tied=tied)

        # Initialize hidden states
        h_init = np.random.randn(hidden_dim)
        h1 = h_init.copy()
        h2 = h_init.copy()

        # Generate inputs
        x1 = np.random.randn(n_steps, hidden_dim) * np.sqrt(R)
        x2 = np.zeros_like(x1)
        x2[:10] = np.random.randn(10, hidden_dim) * np.sqrt(R)  # Uncorrelated
        x2[10:] = x1[10:]  # Perfectly correlated

        for t in range(n_steps):
            h1 = rnn.forward(x1[t], h1, t)
            h2 = rnn.forward(x2[t], h2, t)

            # Compute cosine similarity
            similarities[t] += np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2))

    return similarities / n_trials


# Run experiments
mu_b_values = [-3.0, 0.0, 2.0, 3.5, 5.5]
colors = ['blue', 'darkblue', 'purple', 'brown', 'red']
t = np.arange(100)

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot tied weights
for mu_b, color in zip(mu_b_values, colors):
    print(f"Running tied weights simulation for μb = {mu_b}")
    similarities = run_simulation(mu_b=mu_b, tied=True)
    ax1.plot(t, similarities, color=color, linestyle='-', label=f'μb = {mu_b}')

    # Add theoretical predictions
    theory = [theoretical_prediction(1.0, mu_b, 6.88, tt,
                                     Sigma_12=1.0 if tt >= 10 else 0.0) for tt in t]
    ax1.plot(t, theory, color=color, linestyle='--', alpha=0.5)

ax1.set_xlabel('t')
ax1.set_ylabel('ct')
ax1.set_title('Tied Weights')
ax1.legend()
ax1.grid(True, alpha=0.2)
ax1.set_ylim(0, 1)

# Plot untied weights
for mu_b, color in zip(mu_b_values, colors):
    print(f"Running untied weights simulation for μb = {mu_b}")
    similarities = run_simulation(mu_b=mu_b, tied=False)
    ax2.plot(t, similarities, color=color, linestyle='-', label=f'μb = {mu_b}')

    # Add theoretical predictions
    theory = [theoretical_prediction(1.0, mu_b, 6.88, tt,
                                     Sigma_12=1.0 if tt >= 10 else 0.0) for tt in t]
    ax2.plot(t, theory, color=color, linestyle='--', alpha=0.5)

ax2.set_xlabel('t')
ax2.set_ylabel('ct')
ax2.set_title('Untied Weights')
ax2.legend()
ax2.grid(True, alpha=0.2)
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.show()