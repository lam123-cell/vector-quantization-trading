import numpy as np


class TurboQuant:

    def __init__(self, n_clusters=8, dim=5, lr=0.05):
        self.n_clusters = n_clusters
        self.dim = dim
        self.lr = lr

        self.codebook = None
        self.usage = np.zeros(n_clusters)
        self.is_initialized = False

    def initialize(self, data):
        data = np.array(data)

        indices = np.random.choice(len(data), self.n_clusters, replace=False)
        self.codebook = data[indices]

        self.is_initialized = True
        print("[*] TurboQuant initialized")

    def encode(self, x):
        if not self.is_initialized:
            return None

        x = np.array(x)

        distances = np.linalg.norm(self.codebook - x, axis=1)
        state = int(np.argmin(distances))

        # online update
        self.codebook[state] += self.lr * (x - self.codebook[state])
        self.usage[state] += 1

        return state