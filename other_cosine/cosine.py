import numpy as np
import matplotlib.pyplot as plt


class MinimalRNN:
    def __init__(self, hidden_dim, sigma_w=6.88, sigma_v=1.39, R=0.46, mu_b=0.0):
        self.hidden_dim = hidden_dim
        scale_w = sigma_w / np.sqrt(hidden_dim)
        scale_v = sigma_v / np.sqrt(hidden_dim)
        self.W = np.random.randn(hidden_dim, hidden_dim) * scale_w
        self.V = np.random.randn(hidden_dim, hidden_dim) * scale_v
        self.b = np.full(hidden_dim, mu_b)

    def forward(self, x, h):
        e = np.dot(self.W, h) + np.dot(self.V, x) + self.b
        u = 1 / (1 + np.exp(-e))
        h_new = u * h + (1 - u) * x
        return h_new


def run_simulation(n_steps=100, n_trials=1000, hidden_dim=512, mu_b=-3.0):
    similarities = np.zeros(n_steps)
    R = 0.46

    for trial in range(n_trials):
        rnn = MinimalRNN(hidden_dim, mu_b=mu_b)

        # Initialize both hidden states to be THE SAME
        h_init = np.random.randn(hidden_dim)
        h1 = h_init.copy()  # Start with identical hidden states
        h2 = h_init.copy()

        # Generate all inputs upfront
        x1 = np.random.randn(n_steps, hidden_dim) * np.sqrt(R)
        x2 = np.zeros_like(x1)

        # First 10 steps: uncorrelated inputs
        x2[:10] = np.random.randn(10, hidden_dim) * np.sqrt(R)
        # After 10 steps: perfectly correlated
        x2[10:] = x1[10:]

        for t in range(n_steps):
            h1 = rnn.forward(x1[t], h1)
            h2 = rnn.forward(x2[t], h2)

            # Compute cosine similarity
            similarities[t] += np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2))

    return similarities / n_trials


# Run simulations
mu_b_values = [-3.0, 0.0, 2.0, 3.5, 5.5]
colors = ['blue', 'darkblue', 'purple', 'brown', 'red']
t = np.arange(100)

plt.figure(figsize=(10, 6))
for mu_b, color in zip(mu_b_values, colors):
    print(mu_b)
    similarities = run_simulation(mu_b=mu_b, hidden_dim=512)
    plt.plot(t, similarities, color=color, label=f'μb = {mu_b}')

plt.xlabel('t')
plt.ylabel('ct')
plt.legend()
plt.ylim(0, 1)
plt.grid(True, alpha=0.2)
plt.title('Cosine Similarity Evolution in MinimalRNN')
plt.show()