# bandits.py

import numpy as np

class Bandit:
    '''This class describes a Bandit with n possible actions'''

    BANDIT_TYPES = ("Bernoulli", "Gaussian")

    def __init__(self, n: int, type: str) -> None: 
        assert type in Bandit.BANDIT_TYPES
        self.N = n
        self.type = type
        if type == "Bernoulli":
            # bandits will give reward/no reward
            self.probs = np.random.uniform(size=(n,))
            self.optimal = np.max(self.probs)
        elif type == "Gaussian":
            self.means = np.random.normal(1, 0.2, (n,))
            self.optimal = np.max(self.means)
        self.regret = 0.0

    def get_regret(self) -> float:
        return self.regret

    def getN(self) -> int:
        return self.N

    def reset_regret(self) -> None:
        self.regret = 0.0

    def choose(self, k: int) -> int:
        assert 0 <= k < self.N
        if self.type == "Bernoulli":
            reward = np.random.binomial(1, self.probs[k])
            self.regret += self.optimal - reward
            return reward
        elif self.type == "Gaussian":
            reward = np.random.normal(self.means[k], 1)
            self.regret += self.optimal - reward
            return reward
