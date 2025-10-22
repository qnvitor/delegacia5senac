# ==========================================================
# Alocação de patrulha (Grid + Q-learning)
# ==========================================================
import numpy as np
import random

class GridEnv:
    """Ambiente simples: grid com reward baseado em 'crime probability' por célula."""
    def __init__(self, grid_probs):
        self.grid = grid_probs
        self.n_rows, self.n_cols = self.grid.shape
        self.state = (0,0)

    def reset(self):
        self.state = (0,0)
        return self.state

    def step(self, action):
        # actions: 0=up,1=right,2=down,3=left
        r,c = self.state
        if action==0: r = max(0,r-1)
        if action==1: c = min(self.n_cols-1,c+1)
        if action==2: r = min(self.n_rows-1,r+1)
        if action==3: c = max(0,c-1)
        self.state = (r,c)
        # reward: inverse of crime prob (we want reduce crime) -> reward high if cell risk low after patrolling
        reward = -self.grid[r,c]  # negative of crime prob (minimize)
        done = False
        return self.state, reward, done, {}

def q_learning(grid_probs, episodes=500, alpha=0.1, gamma=0.9, eps=0.2):
    env = GridEnv(grid_probs)
    Q = np.zeros((env.n_rows, env.n_cols, 4))
    for ep in range(episodes):
        s = env.reset()
        for t in range(50):
            if random.random() < eps:
                a = random.randint(0,3)
            else:
                a = np.argmax(Q[s[0], s[1]])
            s2, r, done, _ = env.step(a)
            Q[s[0], s[1], a] = (1-alpha)*Q[s[0], s[1], a] + alpha*(r + gamma*np.max(Q[s2[0], s2[1]]))
            s = s2
    return Q
