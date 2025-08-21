# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 23:14:10 2025

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %% Kalman intro
# Initial belief (mean and covariance)
mu = np.array([0, 0])  # Initial state estimate
Sigma = np.array([[1, 0], [0, 1]])  # Initial uncertainty (covariance matrix)

# Motion model (small random walk)
F = np.eye(2)  # Identity transition
Q = np.array([[0.1, 0], [0, 0.1]])  # Process noise

# Observation model (noisy GPS)
H = np.eye(2)  # Direct observation of (x, y)
R = np.array([[0.5, 0], [0, 0.5]])  # Observation noise

# Generate a simulated true trajectory
true_states = [np.array([0, 0])]
observations = []

# np.random.seed(42)
for _ in range(20):
    # Move randomly
    new_state = true_states[-1] + np.random.multivariate_normal([0.2, 0.2], Q)
    true_states.append(new_state)

    # Noisy observation
    obs = new_state + np.random.multivariate_normal([0, 0], R)
    observations.append(obs)

true_states = np.array(true_states)
observations = np.array(observations)

# Kalman filter update loop
estimated_states = []
for obs in observations:
    # Prediction
    mu = F @ mu
    Sigma = F @ Sigma @ F.T + Q
    
    # Update step
    K = Sigma @ H.T @ np.linalg.inv(H @ Sigma @ H.T + R)
    mu = mu + K @ (obs - H @ mu)
    Sigma = (np.eye(2) - K @ H) @ Sigma
    
    estimated_states.append(mu)

estimated_states = np.array(estimated_states)

# Plot results
plt.figure(figsize=(6,6))
plt.plot(true_states[:,0], true_states[:,1], 'g-', label="True Path")
plt.scatter(observations[:,0], observations[:,1], c='r', marker='x', label="Noisy Observations")
plt.plot(estimated_states[:,0], estimated_states[:,1], 'b--', label="Estimated Path")

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("2D Navigation with Gaussian Belief")
plt.grid()
plt.show()


# %% test with trajectory
# Define 2D actions
actions = {'left': (-1, 0), 'right': (1, 0), 'up': (0, 1), 'down': (0, -1)}

# State transition model
def transition_model(state, action):
    """Returns new state after taking action."""
    dx, dy = actions[action]
    new_state = (state[0] + dx, state[1] + dy)
    return new_state

# Observation model (Noisy observation of current state)
def observation_model(state, noise_std=0.5):
    """Returns a noisy observation of the state."""
    return np.array([
        [state[0] + np.random.normal(0, noise_std)], 
        [state[1] + np.random.normal(0, noise_std)]
    ])  # Shape (2,1)

# Kalman Filter function
def kalman_filter(mu, Sigma, action, observation):
    """
    mu: Mean of belief distribution (2D)
    Sigma: Covariance of belief distribution (2x2)
    action: Chosen action
    observation: Noisy observation (2D)
    """
    # State transition model (Identity dynamics)
    A = np.eye(2)
    B = np.eye(2)  # Action mapped directly
    control_noise = 0.1 * np.eye(2)

    # Prediction step
    action_vec = np.array([[actions[action][0]], [actions[action][1]]])
    mu_pred = A @ mu.reshape(-1, 1) + B @ action_vec
    Sigma_pred = A @ Sigma @ A.T + control_noise  # Increase uncertainty

    # Observation model
    H = np.eye(2)  # Direct observation
    R = 0.2 * np.eye(2)  # Observation noise

    # Kalman Gain
    K = Sigma_pred @ H.T @ np.linalg.inv(H @ Sigma_pred @ H.T + R)

    # Compute residual (observation mismatch)
    residual = observation - (H @ mu_pred)

    # Update mean and covariance
    mu_new = mu_pred + K @ residual
    Sigma_new = (np.eye(2) - K @ H) @ Sigma_pred

    return mu_new.flatten(), Sigma_new

# Simulate trajectory
def simulate_trajectory(start_state, actions_sequence):
    """Simulates a trajectory using Kalman filtering in 2D."""
    mu = np.array(start_state, dtype=float)  # Initial belief mean
    Sigma = 0.1 * np.eye(2)  # Initial covariance

    true_states = [start_state]  # True trajectory
    observed_states = []  # Observed (noisy) trajectory
    estimated_states = [mu.copy()]  # Belief estimates

    for action in actions_sequence:
        # Transition
        new_state = transition_model(true_states[-1], action)
        true_states.append(new_state)

        # Noisy Observation
        observation = observation_model(new_state)
        observed_states.append(observation.flatten())

        # Kalman Filter Update
        mu, Sigma = kalman_filter(mu, Sigma, action, observation)
        estimated_states.append(mu.copy())

    return np.array(true_states), np.array(observed_states), np.array(estimated_states)

# Define action sequence
actions_sequence = ['right', 'right', 'up', 'up', 'left', 'left', 'down', 'right']

# Run simulation
start_state = (0, 0)
true_states, observed_states, estimated_states = simulate_trajectory(start_state, actions_sequence)

# Plot results
plt.figure(figsize=(6, 6))

# Plot true trajectory
plt.plot(true_states[:, 0], true_states[:, 1], 'bo-', label='True Position')

# Plot observations
plt.scatter(observed_states[:, 0], observed_states[:, 1], c='red', alpha=0.5, label='Noisy Observations')

# Plot estimated trajectory
plt.plot(estimated_states[:, 0], estimated_states[:, 1], 'g*-', label='Estimated Position (Belief)')

# Formatting
plt.legend()
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Kalman Filter for 2D Navigation with Discrete Actions")
plt.grid()
plt.show()

# %% POMDP
# transition T(s'|s,a) #### behavioral motion via strategy
# observation P(o|s)   #### odor encounter
# beleif b(s)   #### brain estimate
# policy P(a|b)   #### behavioral strategy
# reward R(s,a) #### RL learning signal
# Q(s,a) #### intermediate table for Q-learning

# %% discrtete belief state
# Define the grid size and goal location
GRID_SIZE = 10
GOAL = (9, 9)  # Goal location in the grid

# Define 2D actions and their effects
ACTIONS = {'left': (-1, 0), 'right': (1, 0), 'up': (0, 1), 'down': (0, -1)}
ACTION_LIST = list(ACTIONS.keys())

# Transition model T(s'|s,a)
def transition_model(state, action):
    """Returns the next state based on the action."""
    x, y = state
    dx, dy = ACTIONS[action]
    x_new, y_new = min(max(x + dx, 0), GRID_SIZE - 1), min(max(y + dy, 0), GRID_SIZE - 1)
    return (x_new, y_new)

# Observation model P(o|s)
def observation_model(state, goal=GOAL, proximity_factor=.8):
    """
    Binary observation: 
    - o=1 (good signal) if close to the goal (higher probability).
    - o=0 (bad signal) otherwise.
    """
    distance = np.linalg.norm(np.array(state) - np.array(goal))
    prob_o1 = np.exp(-distance) * proximity_factor  # Higher probability when closer
    return 1 if np.random.rand() < prob_o1 else 0

# Initialize belief state b(s)
def initialize_belief(grid_size):
    """Initialize a uniform belief distribution."""
    belief = np.ones((grid_size, grid_size)) / (grid_size * grid_size)
    return belief

# Update belief using Bayes' Rule
def update_belief(belief, action, observation):
    """Update the belief distribution using Bayesian filtering."""
    new_belief = np.zeros_like(belief)

    # Prediction step: propagate belief using the transition model
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for a in ACTION_LIST:
                prev_x, prev_y = transition_model((x, y), a)
                new_belief[x, y] += belief[prev_x, prev_y] / len(ACTION_LIST)  # Uniform transition probability
    
    # Correction step: incorporate binary observation
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            prob_o1 = np.exp(-np.linalg.norm(np.array([x, y]) - np.array(GOAL))) * 0.8
            likelihood = prob_o1 if observation == 1 else (1 - prob_o1)
            new_belief[x, y] *= likelihood

    new_belief /= new_belief.sum()  # Normalize belief

    return new_belief

# Action selection policy P(a|b)
def select_action_from_belief(belief):
    """Select an action probabilistically based on belief proximity to goal."""
    expected_x = np.sum(np.arange(GRID_SIZE)[:, None] * belief)
    expected_y = np.sum(np.arange(GRID_SIZE) * belief)

    # Compute direction to goal
    dx = np.sign(GOAL[0] - expected_x)
    dy = np.sign(GOAL[1] - expected_y)

    action_probs = np.array([0.0] * len(ACTION_LIST))
    for i, action in enumerate(ACTION_LIST):
        if ACTIONS[action] == (dx, 0):  # Horizontal movement
            action_probs[i] = 0.5
        if ACTIONS[action] == (0, dy):  # Vertical movement
            action_probs[i] += 0.5

    action_probs /= action_probs.sum()  # Normalize
    return np.random.choice(ACTION_LIST, p=action_probs)

# Simulate the POMDP
def simulate_pomdp(steps=10):
    """Runs a POMDP simulation with binary observations and belief-based action selection."""
    state = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
    belief = initialize_belief(GRID_SIZE)

    plt.figure(figsize=(10, 5))

    for t in range(steps):
        action = select_action_from_belief(belief)
        state = transition_model(state, action)
        observation = observation_model(state)

        belief = update_belief(belief, action, observation)

        # Plot belief map
        plt.subplot(2, steps // 2, t + 1)
        plt.imshow(belief, cmap="Blues", origin="lower")
        plt.scatter(state[1], state[0], color="red", marker="x", label="True State")
        plt.scatter(GOAL[1], GOAL[0], color="green", marker="*", label="Goal")
        plt.xticks([])
        plt.yticks([])
        plt.title(f"Step {t+1}")

    plt.legend()
    plt.tight_layout()
    plt.show()

# Run the simulation
simulate_pomdp(steps=10)

# %% adding QMDP learning to the grid  ################### WORKS ###########################
# Define grid size and goal location
GRID_SIZE = 10
GOAL = (9, 9)  # The agent does NOT know this directly

# Define 2D actions and their effects
ACTIONS = {'left': (-1, 0), 'right': (1, 0), 'up': (0, 1), 'down': (0, -1)}
ACTION_LIST = list(ACTIONS.keys())
NUM_ACTIONS = len(ACTION_LIST)

# Q-learning parameters
ALPHA = 0.2  # Learning rate
GAMMA = 0.98  # Discount factor
EPSILON = 0.1  # Exploration probability
BETA = .1  # temperature

# Initialize Q-table for state-action pairs
Q_table = np.zeros((GRID_SIZE, GRID_SIZE, NUM_ACTIONS))

# Transition model T(s'|s,a)
def transition_model(state, action):
    """Returns the next state based on the action."""
    x, y = state
    dx, dy = ACTIONS[action]
    x_new, y_new = min(max(x + dx, 0), GRID_SIZE - 1), min(max(y + dy, 0), GRID_SIZE - 1)
    return (x_new, y_new)

# Observation model P(o | s)
def observation_model(state, goal=GOAL, lambda_factor=0.05):
    """Returns the probability of detecting the goal based on distance."""
    distance = np.linalg.norm(np.array(state) - np.array(goal))
    return np.exp(-lambda_factor * distance)  # Corrected probability model

# Initialize belief state b(s)
def initialize_belief(grid_size):
    """Initialize a uniform belief distribution over all grid locations."""
    belief = np.ones((grid_size, grid_size)) / (grid_size * grid_size)
    return belief

# Update belief using Bayes' Rule with continuous probability observations
def update_belief(belief, action, observation_prob):
    """Update the belief distribution using Bayesian filtering."""
    new_belief = np.zeros_like(belief)

    # --- 1. Prediction Step: Apply transition model ---
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            prev_x, prev_y = transition_model((x, y), action)  # Use agent's actual action
            new_belief[x, y] += belief[prev_x, prev_y]

    # --- 2. Correction Step: Incorporate observation probability ---
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            likelihood = observation_model((x, y))  # Directly use probability from observation model
            new_belief[x, y] *= likelihood  

    # Normalize belief (Avoid division by zero)
    total_belief = np.sum(new_belief)
    if total_belief > 0:
        new_belief /= total_belief
    else:
        new_belief = np.ones_like(new_belief) / (GRID_SIZE * GRID_SIZE)  # Reset to uniform belief if numerical error

    return new_belief

# Reward function R(s)
def reward_function(state):
    """Returns reward: +1 for reaching the goal, otherwise 0."""
    return 1.0 if np.array_equal(state, GOAL) else 0.0

# Q-learning update
def update_q_table(state, action, reward, next_state):
    """Update Q-table using Q-learning."""
    x, y = state
    next_x, next_y = next_state
    action_idx = ACTION_LIST.index(action)

    best_next_q = np.max(Q_table[next_x, next_y])  # Best Q-value for next state

    # Q-learning update with more structured weight propagation
    Q_table[x, y, action_idx] += ALPHA * (reward + GAMMA * best_next_q - Q_table[x, y, action_idx])

    # Keep Q-values within reasonable range
    Q_table[x, y, action_idx] = np.clip(Q_table[x, y, action_idx], -1, 1)


# Select action using belief-weighted Q-learning policy
def select_action_from_belief(belief):
    """Select an action using epsilon-greedy strategy over belief-weighted Q-values."""
    
    if np.random.rand() < EPSILON:
        return np.random.choice(ACTION_LIST)  # Random exploration

    action_values = np.zeros(NUM_ACTIONS)

    # Compute expected Q-value for each action over the belief distribution
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for a in range(NUM_ACTIONS):
                action_values[a] += Q_table[x, y, a] * belief[x, y]

    # Force stronger directionality
    action_values -= np.mean(action_values)  # Remove baseline bias
    action_values = np.exp(action_values / BETA)  # Sharpen selection (temperature=0.1)

    action_probs = action_values / np.sum(action_values)  # Normalize

    return np.random.choice(ACTION_LIST, p=action_probs)

# Simulate the POMDP with Q-learning and track rewards
def train_q_learning(episodes=500, steps=30):
    """Trains the Q-learning policy over multiple episodes and tracks rewards."""
    total_rewards = []

    for episode in range(episodes):
        state = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
        belief = initialize_belief(GRID_SIZE)
        episode_reward = 0  # Track total reward per episode

        for _ in range(steps):
            action = select_action_from_belief(belief)
            next_state = transition_model(state, action)
            observation_prob = observation_model(next_state)  

            belief = update_belief(belief, action, observation_prob)
            reward = reward_function(next_state)
            episode_reward += reward  # Accumulate reward

            update_q_table(state, action, reward, next_state)
            state = next_state

            if np.array_equal(state, GOAL):  # Stop if goal reached
                break

        total_rewards.append(episode_reward)  # Store reward for analysis

    # Plot reward progression
    plt.figure(figsize=(8, 4))
    plt.plot(total_rewards, '-o', label="Total Reward per Episode", alpha=0.7)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning Training Progress")
    plt.legend()
    plt.grid()
    plt.show()

# Test the trained policy
def simulate_pomdp_with_q_policy(steps=10):
    """Runs a POMDP simulation with the learned belief-weighted Q-policy."""
    state = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
    belief = initialize_belief(GRID_SIZE)

    plt.figure(figsize=(10, 5))

    for t in range(steps):
        action = select_action_from_belief(belief)
        state = transition_model(state, action)
        observation_prob = observation_model(state)
        belief = update_belief(belief, action, observation_prob)

        # Plot belief map
        plt.subplot(2, steps // 2, t + 1)
        plt.imshow(belief, origin="lower", cmap="Blues")
        plt.scatter(state[1], state[0], color="red", marker="x", label="True State")
        plt.scatter(GOAL[1], GOAL[0], color="green", marker="*", label="Goal")
        plt.xticks([])
        plt.yticks([])
        plt.title(f"Step {t+1}")

        if np.array_equal(state, GOAL):  # Stop if goal reached
            break

    plt.legend()
    plt.tight_layout()
    plt.show()

# Train Q-learning policy and plot rewards
train_q_learning(episodes=500, steps=20)

# Run the simulation using the learned policy
simulate_pomdp_with_q_policy(steps=10)

# show observation field
plt.figure()
obs_prob_grid = np.zeros((GRID_SIZE, GRID_SIZE))
for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        obs_prob_grid[x, y] = observation_model((x, y))

# Plot observation probability heatmap
plt.figure(figsize=(6, 5))
plt.imshow(obs_prob_grid, origin="lower", cmap="Blues")
plt.colorbar(label="Observation Probability")
plt.scatter(GOAL[1], GOAL[0], color="red", marker="*", s=100, label="Goal")
plt.title("Observation Probability Across Grid")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.xticks(range(GRID_SIZE))
plt.yticks(range(GRID_SIZE))
plt.legend()

# %% NEXT ###
# check binary encounter
# check Gaussian approximation
# play with perturbations

# %% continuous space
import numpy as np
import matplotlib.pyplot as plt

# Define the grid size
GRID_SIZE = 10
GOAL = (5,5)

# Define actions and movement
ACTIONS = {'left': (-1, 0), 'right': (1, 0), 'up': (0, 1), 'down': (0, -1)}
ACTION_LIST = list(ACTIONS.keys())

# Transition model T(s'|s,a)
def transition_model(state, action):
    """Returns the next state based on the action."""
    x, y = state
    dx, dy = ACTIONS[action]
    x_new, y_new = min(max(x + dx, 0), GRID_SIZE - 1), min(max(y + dy, 0), GRID_SIZE - 1)
    return (x_new, y_new)

# Observation model P(o|s)
def observation_model(state, goal=GOAL, proximity_factor=.8):
    """
    Binary observation: 
    - o=1 (good signal) if close to the goal (higher probability).
    - o=0 (bad signal) otherwise.
    """
    distance = np.linalg.norm(np.array(state) - np.array(goal))
    prob_o1 = np.exp(-distance) * proximity_factor  # Higher probability when closer
    return 1 if np.random.rand() < prob_o1 else 0

# Initialize belief as a 2D Gaussian
def initialize_belief():
    """Initialize a Gaussian belief with mean at center and high uncertainty."""
    mu = np.array([GRID_SIZE // 2, GRID_SIZE // 2], dtype=float)
    Sigma = np.eye(2) * GRID_SIZE  # Large initial uncertainty
    return mu, Sigma

# Update belief using a 2D Gaussian filter
def update_belief(mu, Sigma, action, observation):
    """Update the Gaussian belief based on action and observation using Kalman-like update."""
    # Motion model (prediction step)
    motion_noise = np.eye(2) * 0.5  # Small uncertainty in motion
    A = np.eye(2)  # Linear transition model
    action_vec = np.array(ACTIONS[action]).reshape(2, 1)
    mu_pred = (A @ mu.reshape(-1, 1) + action_vec).flatten()
    Sigma_pred = A @ Sigma @ A.T + motion_noise  # Update uncertainty

    # Observation model (update step)
    H = np.eye(2)  # Direct observation
    R = np.eye(2) * 0.5  # Observation noise
    K = Sigma_pred @ H.T @ np.linalg.inv(H @ Sigma_pred @ H.T + R)  # Kalman Gain

    # Compute observation residual (assume a noisy estimate of state)
    obs_noise = np.random.multivariate_normal(mu_pred, R) if observation == 1 else np.random.multivariate_normal(mu_pred, R * 2)
    residual = obs_noise.reshape(2, 1) - (H @ mu_pred.reshape(-1, 1))

    # Update Gaussian belief
    mu_new = mu_pred.reshape(-1, 1) + K @ residual
    Sigma_new = (np.eye(2) - K @ H) @ Sigma_pred

    return mu_new.flatten(), Sigma_new

# Action selection policy based only on belief
def select_action_from_belief(mu):
    """Select an action probabilistically based on belief, without knowledge of the goal."""
    # Move towards highest probability area (mean of Gaussian)
    move_probabilities = {a: 0.0 for a in ACTION_LIST}
    for action, (dx, dy) in ACTIONS.items():
        move_vector = np.array([dx, dy])
        dot_product = -1 * np.dot(move_vector, mu)  # Movement alignment with expected state
        move_probabilities[action] = np.exp(dot_product)  # Exponential weighting for probabilistic choice

    # Normalize probabilities
    total_prob = sum(move_probabilities.values())
    move_probabilities = {a: p / total_prob for a, p in move_probabilities.items()}
    
    return np.random.choice(ACTION_LIST, p=list(move_probabilities.values()))

# Simulate POMDP with Gaussian belief
def simulate_pomdp(steps=10):
    """Runs a POMDP simulation with Gaussian belief and belief-based action selection."""
    state = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
    mu, Sigma = initialize_belief()

    plt.figure(figsize=(10, 5))

    for t in range(steps):
        action = select_action_from_belief(mu)
        state = transition_model(state, action)
        observation = observation_model(state)

        mu, Sigma = update_belief(mu, Sigma, action, observation)

        # Plot belief map
        plt.subplot(2, steps // 2, t + 1)
        X, Y = np.meshgrid(np.linspace(0, GRID_SIZE-1, GRID_SIZE), np.linspace(0, GRID_SIZE-1, GRID_SIZE))
        Z = np.exp(-0.5 * (((X - mu[0]) ** 2 / Sigma[0, 0]) + ((Y - mu[1]) ** 2 / Sigma[1, 1])))
        plt.contourf(X, Y, Z, cmap="Blues")

        plt.scatter(state[1], state[0], color="red", marker="x", label="True State")
        plt.scatter(GOAL[1], GOAL[0], color="green", marker="*", label="Goal")
        plt.xticks([])
        plt.yticks([])
        plt.title(f"Step {t+1}")

    plt.legend()
    plt.tight_layout()
    plt.show()

# Run the simulation
simulate_pomdp(steps=10)

# %% continuous space with Q-learning
# Define the grid size and goal location
GRID_SIZE = 10
GOAL = (7, 7)  # The agent does NOT know this directly

# Define actions and movement
ACTIONS = {'left': (-1, 0), 'right': (1, 0), 'up': (0, 1), 'down': (0, -1)}
ACTION_LIST = list(ACTIONS.keys())
NUM_ACTIONS = len(ACTION_LIST)

# Learning parameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.2  # Exploration probability

# Initialize Q-table
Q_table = np.zeros((GRID_SIZE, GRID_SIZE, NUM_ACTIONS))

# Transition model T(s'|s,a)
def transition_model(state, action):
    """Returns the next state based on the action."""
    x, y = state
    dx, dy = ACTIONS[action]
    x_new, y_new = min(max(x + dx, 0), GRID_SIZE - 1), min(max(y + dy, 0), GRID_SIZE - 1)
    return (x_new, y_new)

# Observation model P(o|s) - Higher probability of correct signal near goal
def observation_model(state, goal=GOAL):
    """
    Binary observation:
    - o=1 (good signal) with higher probability when closer to goal.
    - o=0 (bad signal) otherwise.
    """
    distance = np.linalg.norm(np.array(state) - np.array(goal))
    prob_o1 = np.exp(-distance)  # Exponentially higher probability when close
    return 1 if np.random.rand() < prob_o1 else 0  # Binary observation

# Reward function R(s, a)
def reward_function(state):
    """Returns reward for state-action transitions."""
    return 1.0 if state == GOAL else -0.01  # Reward goal, penalize steps

# Initialize belief as a 2D Gaussian
def initialize_belief():
    """Initialize a Gaussian belief with mean at center and high uncertainty."""
    mu = np.array([GRID_SIZE // 2, GRID_SIZE // 2], dtype=float)
    Sigma = np.eye(2) * GRID_SIZE  # Large initial uncertainty
    return mu, Sigma

# Update belief using a 2D Gaussian filter
def update_belief(mu, Sigma, action, observation):
    """Update the Gaussian belief based on action and observation using Kalman-like update."""
    # Motion model (prediction step)
    motion_noise = np.eye(2) * 0.5  # Small uncertainty in motion
    A = np.eye(2)  # Linear transition model
    action_vec = np.array(ACTIONS[action]).reshape(2, 1)
    mu_pred = (A @ mu.reshape(-1, 1) + action_vec).flatten()
    Sigma_pred = A @ Sigma @ A.T + motion_noise  # Update uncertainty

    # Observation model (update step)
    H = np.eye(2)  # Direct observation
    R = np.eye(2) * 0.5  # Observation noise
    K = Sigma_pred @ H.T @ np.linalg.inv(H @ Sigma_pred @ H.T + R)  # Kalman Gain

    # Compute observation residual (if observation is good, assume closer to correct state)
    obs_noise = np.random.multivariate_normal(mu_pred, R) if observation == 1 else np.random.multivariate_normal(mu_pred, R * 2)
    residual = obs_noise.reshape(2, 1) - (H @ mu_pred.reshape(-1, 1))

    # Update Gaussian belief
    mu_new = mu_pred.reshape(-1, 1) + K @ residual
    Sigma_new = (np.eye(2) - K @ H) @ Sigma_pred

    return mu_new.flatten(), Sigma_new

# Q-Learning action selection policy P(a|b)
def q_learning_action_selection(state):
    """
    Select an action using ε-greedy Q-learning policy.
    """
    x, y = state
    if np.random.rand() < EPSILON:  # Explore
        return np.random.choice(ACTION_LIST)
    else:  # Exploit (choose best action from Q-table)
        best_action_idx = np.argmax(Q_table[x, y])
        return ACTION_LIST[best_action_idx]
    
def compute_policy_from_belief(mu, Sigma):
    """Compute policy P(a|b) as a weighted sum of Q(a,s)b(s)."""
    action_values = np.zeros(NUM_ACTIONS)

    # Iterate over possible states (since belief is continuous, approximate via sampling)
    for _ in range(100):  # Sample from the belief
        sampled_state = np.clip(np.random.multivariate_normal(mu, Sigma).astype(int), 0, GRID_SIZE - 1)
        x, y = sampled_state
        action_values += Q_table[x, y]  # Sum over Q(a, s) * b(s)

    # Apply softmax for smooth policy selection
    action_probs = np.exp(action_values)
    action_probs /= np.sum(action_probs)  # Normalize

    return np.random.choice(ACTION_LIST, p=action_probs)

# Update Q-table
def update_q_table(state, action, reward, next_state):
    """
    Update Q-table using the Bellman equation.
    """
    x, y = state
    next_x, next_y = next_state
    action_idx = ACTION_LIST.index(action)

    best_next_q = np.max(Q_table[next_x, next_y])
    Q_table[x, y, action_idx] += ALPHA * (reward + GAMMA * best_next_q - Q_table[x, y, action_idx])

# Simulate POMDP with Gaussian belief and Q-learning
def simulate_pomdp(steps=10, training_episodes=100):
    """Runs a POMDP simulation with Gaussian belief and Q-learning policy."""
    
    # Train the agent using Q-learning
    for episode in range(training_episodes):
        state = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
        for _ in range(steps):
            action = q_learning_action_selection(state)
            next_state = transition_model(state, action)
            reward = reward_function(next_state)
            update_q_table(state, action, reward, next_state)
            state = next_state
            if state == GOAL:
                break  # Stop if goal reached

    # Testing Phase: Run the trained policy
    state = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
    mu, Sigma = initialize_belief()

    plt.figure(figsize=(10, 5))

    for t in range(steps):
        action = q_learning_action_selection(state)
        state = transition_model(state, action)
        observation = observation_model(state)

        mu, Sigma = update_belief(mu, Sigma, action, observation)

        # Plot belief map
        plt.subplot(2, steps // 2, t + 1)
        X, Y = np.meshgrid(np.linspace(0, GRID_SIZE-1, GRID_SIZE), np.linspace(0, GRID_SIZE-1, GRID_SIZE))
        Z = np.exp(-0.5 * (((X - mu[0]) ** 2 / Sigma[0, 0]) + ((Y - mu[1]) ** 2 / Sigma[1, 1])))
        plt.contourf(X, Y, Z, cmap="Blues")

        plt.scatter(state[1], state[0], color="red", marker="x", label="True State")
        plt.scatter(GOAL[1], GOAL[0], color="green", marker="*", label="Goal")
        plt.xticks([])
        plt.yticks([])
        plt.title(f"Step {t+1}")

    plt.legend()
    plt.tight_layout()
    plt.show()

# Run the simulation
simulate_pomdp(steps=10, training_episodes=500)

# %%
import numpy as np
import matplotlib.pyplot as plt

# Define the grid size and goal location
GRID_SIZE = 6
GOAL = (3, 3)  # The agent does NOT know this directly

# Define actions and movement
ACTIONS = {'left': (-1, 0), 'right': (1, 0), 'up': (0, 1), 'down': (0, -1)}
ACTION_LIST = list(ACTIONS.keys())
NUM_ACTIONS = len(ACTION_LIST)

# Learning parameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.2  # Exploration probability

# Initialize Q-table (state-action pairs)
Q_table = np.zeros((GRID_SIZE, GRID_SIZE, NUM_ACTIONS))

# Transition model T(s'|s,a)
def transition_model(state, action):
    """Returns the next state based on the action."""
    x, y = state
    dx, dy = ACTIONS[action]
    x_new, y_new = min(max(x + dx, 0), GRID_SIZE - 1), min(max(y + dy, 0), GRID_SIZE - 1)
    return (x_new, y_new)

# Observation model P(o|s) - Higher probability of correct signal near goal
def observation_model(state, goal=GOAL):
    """Binary observation: Higher probability of correct signal when near goal."""
    distance = np.linalg.norm(np.array(state) - np.array(goal))
    prob_o1 = np.exp(-distance)  # Higher probability when closer to goal
    return 1 if np.random.rand() < prob_o1 else 0  # Binary observation

# Reward function R(s, a)
def reward_function(state):
    """Returns reward for state-action transitions."""
    return 1.0 if np.array_equal(state, GOAL) else -0.01  # Reward goal, penalize steps

# Initialize belief covariance (since mean is always the true state)
def initialize_belief():
    """Initialize belief covariance with high uncertainty."""
    Sigma = np.eye(2) * GRID_SIZE  # Large initial uncertainty
    return Sigma

# Update belief covariance based on motion and observation
def update_belief_covariance(Sigma, observation):
    """Update only the uncertainty (covariance), while keeping belief centered on the true state."""
    motion_noise = np.eye(2) * 0.5  # Small uncertainty in motion
    Sigma_pred = Sigma + motion_noise  # Increase uncertainty over time

    # Observation model update (reduces uncertainty)
    H = np.eye(2)  # Direct observation
    R = np.eye(2) * 0.5  # Observation noise
    K = Sigma_pred @ H.T @ np.linalg.inv(H @ Sigma_pred @ H.T + R)  # Kalman Gain

    # Update belief covariance
    Sigma_new = (np.eye(2) - K @ H) @ Sigma_pred

    return Sigma_new

# Compute action probabilities P(a | b) using max(Q(a,s) * b(s))
def compute_policy_from_belief(state, Sigma):
    """Compute policy P(a|b) as max_s Q(a, s) b(s)."""
    belief_grid = np.zeros((GRID_SIZE, GRID_SIZE))

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            diff = np.array([x, y]) - state
            belief_grid[x, y] = np.exp(-0.5 * (diff.T @ np.linalg.inv(Sigma) @ diff))

    belief_grid /= np.sum(belief_grid)  # Normalize belief

    # Compute Q(a, s) * b(s)
    action_values = np.zeros(NUM_ACTIONS)
    for a in range(NUM_ACTIONS):
        action_values[a] = np.max(Q_table[:, :, a] * belief_grid)  # Max over weighted Q-values

    # Apply softmax for smooth policy selection
    action_probs = np.exp(action_values)
    action_probs /= np.sum(action_probs)

    return np.random.choice(ACTION_LIST, p=action_probs)

# Q-learning update for belief
def update_q_table(state, Sigma, action, reward, next_state, next_Sigma):
    """Update Q-table using belief-based learning."""
    x, y = state
    next_x, next_y = next_state
    action_idx = ACTION_LIST.index(action)

    best_next_q = np.max(Q_table[next_x, next_y])
    Q_table[x, y, action_idx] += ALPHA * (reward + GAMMA * best_next_q - Q_table[x, y, action_idx])

# Simulate POMDP with visualization in subplot format
def simulate_pomdp(steps=10, training_episodes=500):
    """Runs a POMDP simulation with Gaussian belief covariance tracking."""

    # Train the Q-table
    for episode in range(training_episodes):
        state = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
        Sigma = initialize_belief()

        for _ in range(steps):
            action = compute_policy_from_belief(state, Sigma)
            next_state = transition_model(state, action)
            observation = observation_model(next_state)
            next_Sigma = update_belief_covariance(Sigma, observation)
            reward = reward_function(next_state)
            update_q_table(state, Sigma, action, reward, next_state, next_Sigma)
            state, Sigma = next_state, next_Sigma
            if np.array_equal(state, GOAL):
                break

    # Testing Phase: Run the trained policy
    state = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
    Sigma = initialize_belief()

    plt.figure(figsize=(10, 5))

    for t in range(steps):
        action = compute_policy_from_belief(state, Sigma)
        state = transition_model(state, action)
        observation = observation_model(state)
        Sigma = update_belief_covariance(Sigma, observation)
        
        # print(Sigma)
        if np.array_equal(state, GOAL):  # End episode if goal is reached
            break

        plt.subplot(2, steps // 2, t + 1)
        X, Y = np.meshgrid(np.linspace(0, GRID_SIZE-1, GRID_SIZE), np.linspace(0, GRID_SIZE-1, GRID_SIZE))
        belief_center = np.array(state)  # **Corrected: belief centered at true state**
        Z = np.exp(-0.5 * (((X - belief_center[0]) ** 2 / Sigma[0, 0]) + ((Y - belief_center[1]) ** 2 / Sigma[1, 1])))

        plt.contourf(X, Y, Z, cmap="Blues", alpha=0.6)
        plt.scatter(state[1], state[0], color="red", marker="x", label="True State (Belief Center)")
        plt.scatter(GOAL[1], GOAL[0], color="green", marker="*", label="Goal")

        plt.xticks(range(GRID_SIZE))
        plt.yticks(range(GRID_SIZE))
        plt.legend()

    plt.tight_layout()
    plt.show()

# Run the simulation
simulate_pomdp(steps=10, training_episodes=500)

# %%
###############################################################################
# next: think about inference of B(s)


