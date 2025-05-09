# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 15:02:48 2025

@author: ksc75
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# %%
# ----------------------------------------------------------
# Environment setup
class SimpleEnvironmentStochastic:
    def __init__(self, epsilon=0.3):
        self.state = np.random.choice([0, 1])  # 0 = down, 1 = up
        self.epsilon = epsilon  # misalignment probability

    def step(self, action):
        """
        Action: 0 = turn, 1 = continue
        """
        reward = 0
        if self.state == 1 and action == 1:
            reward = +1  # Good to continue when up
        elif self.state == 0 and action == 1:
            reward = -1  # Bad to continue when down
        else:
            reward = 0  # Neutral reward for turning

        # Environment dynamics
        if action == 0:
            # If turn -> randomize environment state
            self.state = np.random.choice([0, 1])
        elif action == 1:
            # If continue -> small chance of misalignment (flip)
            if np.random.rand() < self.epsilon:
                self.state = 1 - self.state  # flip state

        return self.state, reward
    
    def reset(self):
        self.state = np.random.choice([0, 1])

# ----------------------------------------------------------
# Policy model
class SimplePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Two parameters: prob of "continue" (action=1) in state 0 and 1
        self.logits = nn.Parameter(torch.zeros(2))  # logits for state 0 and state 1

    def forward(self, state):
        """
        Return action distribution given state.
        """
        logit = self.logits[state]  # Pick corresponding logit
        prob_continue = torch.sigmoid(logit)  # probability of action=1 ("continue")
        probs = torch.stack([1 - prob_continue, prob_continue])  # [p(turn), p(continue)]
        return probs

# ----------------------------------------------------------
# Training function
def train_policy_stochastic(num_episodes=500, gamma=0.9, epsilon=0.1):
    env = SimpleEnvironmentStochastic(epsilon=epsilon)
    policy = SimplePolicy()
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    all_rewards = []

    for episode in range(num_episodes):
        log_probs = []
        rewards = []

        env.reset()

        for t in range(20):
            state = env.state
            probs = policy(state)

            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)

            next_state, reward = env.step(action.item())

            log_probs.append(log_prob)
            rewards.append(reward)

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        for log_prob, Gt in zip(log_probs, returns):
            policy_loss.append(-log_prob * Gt)
        policy_loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        all_rewards.append(np.sum(rewards))

        if episode % 50 == 0 or episode == num_episodes-1:
            print(f"Episode {episode}, Total Reward = {np.sum(rewards)}")

    return policy, all_rewards

# ----------------------------------------------------------
# Simulate Trajectories
def simulate_trajectory(policy, epsilon=0.1, random_policy=False):
    env = SimpleEnvironmentStochastic(epsilon=epsilon)
    env.reset()

    x = 0.0  # initial position
    positions = [x]

    for t in range(30):
        state = env.state
        if random_policy:
            action = np.random.choice([0, 1])
        else:
            probs = policy(state)
            m = torch.distributions.Categorical(probs)
            action = m.sample().item()

        if action == 1:  # continue
            if state == 1:
                x += 1.0
            else:
                x -= 2.0
        # if action==0 (turn), x stays the same, just flips internal state xxx
        if action == 0:
            dx = np.random.choice([-1, 1])
            x += dx

        env.step(action)
        positions.append(x)

    return positions

# ----------------------------------------------------------
# Main Execution
# if __name__ == "__main__":
epsilon = 0.1
policy, all_rewards = train_policy_stochastic(num_episodes=500, epsilon=epsilon)

# ----------------------------------------------------------
# Plot learning curve
plt.plot(all_rewards)
plt.xlabel('Episode')
plt.ylabel('Total reward per episode')
plt.title('Policy learning over episodes')
plt.grid()
plt.show()

# Simulate trajectories
traj_trained = simulate_trajectory(policy, epsilon=epsilon, random_policy=False)
traj_random = simulate_trajectory(policy, epsilon=epsilon, random_policy=True)

# Plot trajectories
plt.figure(figsize=(8,5))
plt.plot(traj_random, label='Random Policy', linestyle='--')
plt.plot(traj_trained, label='Trained Policy', linestyle='-')
plt.xlabel('Time step')
plt.ylabel('Position')
plt.title('Position Trajectories: Random vs Trained')
plt.legend()
plt.grid()
plt.show()

# ----------------------------------------------------------
# Inspect learned policy
print("\nLearned probabilities:")
for s in [0, 1]:
    probs = policy(s).detach().numpy()
    print(f"State {s}: Turn = {probs[0]:.2f}, Continue = {probs[1]:.2f}")
