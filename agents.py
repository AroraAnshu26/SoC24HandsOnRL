# agents.py

import numpy as np
from bandits import Bandit

class Agent:
    def __init__(self, bandit: Bandit) -> None:
        self.bandit = bandit
        self.banditN = bandit.getN()
        self.rewards = 0
        self.numiters = 0
    
    def action(self) -> int:
        raise NotImplementedError()

    def update(self, choice: int, reward: int) -> None:
        raise NotImplementedError()

    def act(self) -> int:
        choice = self.action()
        reward = self.bandit.choose(choice)
        self.rewards += reward
        self.numiters += 1
        self.update(choice, reward)
        return reward

class GreedyAgent(Agent):
    def __init__(self, bandit: Bandit, initialQ: float) -> None:
        super().__init__(bandit)
        self.Q = np.full(bandit.getN(), initialQ)
        self.N = np.zeros(bandit.getN())

    def action(self) -> int:
        return np.argmax(self.Q)

    def update(self, choice: int, reward: int) -> None:
        self.N[choice] += 1
        self.Q[choice] += (reward - self.Q[choice]) / self.N[choice]

class epsGreedyAgent(Agent):
    def __init__(self, bandit: Bandit, epsilon: float) -> None:
        super().__init__(bandit)
        self.epsilon = epsilon
        self.Q = np.zeros(bandit.getN())
        self.N = np.zeros(bandit.getN())

    def action(self) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.banditN)
        else:
            return np.argmax(self.Q)

    def update(self, choice: int, reward: int) -> None:
        self.N[choice] += 1
        self.Q[choice] += (reward - self.Q[choice]) / self.N[choice]

class UCBAAgent(Agent):
    def __init__(self, bandit: Bandit, c: float) -> None:
        super().__init__(bandit)
        self.c = c
        self.Q = np.zeros(bandit.getN())
        self.N = np.zeros(bandit.getN())
        self.t = 0

    def action(self) -> int:
        self.t += 1
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / (self.N + 1e-5))
        return np.argmax(ucb_values)

    def update(self, choice: int, reward: int) -> None:
        self.N[choice] += 1
        self.Q[choice] += (reward - self.Q[choice]) / self.N[choice]

class GradientBanditAgent(Agent):
    def __init__(self, bandit: Bandit, alpha: float) -> None:
        super().__init__(bandit)
        self.alpha = alpha
        self.H = np.zeros(bandit.getN())
        self.avg_reward = 0

    def action(self) -> int:
        exp_H = np.exp(self.H)
        self.probs = exp_H / np.sum(exp_H)
        return np.random.choice(np.arange(self.banditN), p=self.probs)

    def update(self, choice: int, reward: int) -> None:
        self.avg_reward += (reward - self.avg_reward) / self.numiters
        self.H -= self.alpha * (reward - self.avg_reward) * self.probs
        self.H[choice] += self.alpha * (reward - self.avg_reward)

class ThompsonSamplerAgent(Agent):
    def __init__(self, bandit: Bandit) -> None:
        super().__init__(bandit)
        self.alpha = np.ones(bandit.getN())
        self.beta = np.ones(bandit.getN())

    def action(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, choice: int, reward: int) -> None:
        if reward == 1:
            self.alpha[choice] += 1
        else:
            self.beta[choice] += 1
