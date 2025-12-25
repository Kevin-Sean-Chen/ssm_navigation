# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:40:21 2024

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt

# %%
import numpy as np
import matplotlib.pyplot as plt

# Define the environment (target point in 2D space)
target = np.array([100, 100])  # Target coordinates (x, y)

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor

# Define the number of states and actions
num_states = 3  # Hidden states from the AR-HMM
num_actions = 4  # Actions could be moving to neighboring hidden states

# Initialize Q-table: state x action
Q_table = np.zeros((num_states, num_actions))

# Function to compute reward based on distance to target
def compute_reward(position, target):
    distance = np.linalg.norm(position - target)
    return -distance  # Negative reward for being far from the target

# Function to select an action using epsilon-greedy policy
def select_action(state, Q_table, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)  # Exploration
    else:
        return np.argmax(Q_table[state])  # Exploitation

# Function to update the agent's position based on AR-HMM emissions (velocity)
def update_position(position, emission):
    return position + emission

# Function to run Q-learning for navigation
def q_learning_arhmm(transition_matrix, means, std_devs, weights, num_episodes, seq_length):
    num_states = transition_matrix.shape[0]
    position = np.array([0, 0])  # Starting position (x, y)

    # Initialize epsilon here
    epsilon = 1.0  # Starting value for epsilon
    
    # Track the positions and states for plotting
    positions = []
    state_history = []

    for episode in range(num_episodes):
        # Initialize hidden state from the HMM
        state = np.random.choice(num_states)

        # Initial velocity (emission from the AR-HMM)
        emission_x = np.random.normal(means[state, 0], std_devs[state, 0])
        emission_y = np.random.normal(means[state, 1], std_devs[state, 1])
        emission = np.array([emission_x, emission_y])

        for t in range(seq_length):
            # Select an action (transition between hidden states)
            action = select_action(state, Q_table, epsilon)

            # Apply the selected action by transitioning between states in the AR-HMM
            next_state = np.random.choice(num_states, p=transition_matrix[state, :])

            # Generate new emissions (velocities) from the AR-HMM
            emission_x = means[next_state, 0] + weights[next_state, 0] * emission[0] + np.random.normal(0, std_devs[next_state, 0])
            emission_y = means[next_state, 1] + weights[next_state, 1] * emission[1] + np.random.normal(0, std_devs[next_state, 1])
            emission = np.array([emission_x, emission_y])

            # Update the agent's position based on the new emission (velocity)
            next_position = update_position(position, emission)

            # Store the position and state for plotting
            positions.append(next_position)
            state_history.append(state)

            # Compute the reward based on the new position
            reward = compute_reward(next_position, target)

            # Update Q-value using Q-learning formula
            best_next_action = np.argmax(Q_table[next_state])
            Q_table[state, action] += alpha * (reward + gamma * Q_table[next_state, best_next_action] - Q_table[state, action])

            # Update the current state and position
            state = next_state
            position = next_position

            # If the agent is close to the target, break the episode
            if np.linalg.norm(position - target) < 0.1:
                break

        # Reduce epsilon over time to favor exploitation over exploration
        epsilon = max(epsilon * 0.99, 0.01)  # Ensure epsilon doesn't go below a threshold

    return Q_table, np.array(positions), np.array(state_history)

# Define parameters for the AR-HMM (three states)
transition_matrix = np.array([[0.7, 0.2, 0.1],
                              [0.3, 0.5, 0.2],
                              [0.2, 0.3, 0.5]])

# Mean velocities for each state in 2D (vx, vy)
means = np.array([[1.0, 0.5],  # State 1 mean velocities (vx, vy)
                  [0.5, 1.0],  # State 2 mean velocities (vx, vy)
                  [0.0, 1.5]]) # State 3 mean velocities (vx, vy)

# Standard deviations for each state (separate for vx and vy)
std_devs = np.array([[0.2, 0.1],  # State 1 std deviations (vx_std, vy_std)
                     [0.3, 0.2],  # State 2 std deviations (vx_std, vy_std)
                     [0.1, 0.3]]) # State 3 std deviations (vx_std, vy_std)

# Autoregressive weights for each state (for vx and vy)
weights = np.array([[0.7, 0.5],  # State 1 weights (wx, wy)
                    [0.4, 0.6],  # State 2 weights (wx, wy)
                    [0.6, 0.4]]) # State 3 weights (wx, wy)

# Run Q-learning for AR-HMM
num_episodes = 10
seq_length = 100  # Length of each episode (steps)
Q_table, positions, state_history = q_learning_arhmm(transition_matrix, means, std_devs, weights, num_episodes, seq_length)

# Output the learned Q-table
print("Learned Q-table:")
print(Q_table)

# Ensure positions and states have the same length
if len(positions) != len(state_history):
    print("Warning: Length mismatch between positions and state history. Trimming state history.")
    state_history = state_history[:len(positions)]

# Plot the 2D track of navigation color-coded by states
plt.figure(figsize=(10, 8))
plt.scatter(positions[:, 0], positions[:, 1], c=state_history, cmap='viridis', marker='o', s=50)
plt.colorbar(label='Hidden States')
plt.plot(target[0], target[1], 'r*', markersize=15)  # Mark the target location
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('2D Navigation with Color-Coded States')
plt.grid()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Define the environment (target point in 2D space)
target = np.array([100, 100])  # Target coordinates (x, y)

# Define the number of states and actions
num_states = 3  # Hidden states from the AR-HMM
num_actions = num_states  # Actions are the state transitions

# Function to compute reward based on distance to target
def compute_reward(position, target):
    distance = np.linalg.norm(position - target)
    return -distance  # Negative reward for being far from the target

# Function to update the agent's position based on AR-HMM emissions (velocity)
def update_position(position, emission):
    return position + emission

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 300  # Number of episodes for training
seq_length = 500  # Length of each sequence

# Initialize the Q-table
Q_table = np.zeros((num_states, num_actions))

# Define parameters for the AR-HMM (three states)
transition_matrix = np.array([[0.9, 0.05, 0.05],
                              [0.1, 0.8, 0.1],
                              [0.2, 0.1, 0.7]])

# Mean velocities for each state in 2D (vx, vy)
means = np.array([[1.0, 0.5],  # State 1 mean velocities (vx, vy)
                  [-0.5, -1.0],  # State 2 mean velocities (vx, vy)
                  [1.0, 1.5]]) # State 3 mean velocities (vx, vy)

# Standard deviations for each state (separate for vx and vy)
std_devs = np.array([[0.2, 0.1],  # State 1 std deviations (vx_std, vy_std)
                     [0.3, 0.2],  # State 2 std deviations (vx_std, vy_std)
                     [0.1, 0.3]])*10 # State 3 std deviations (vx_std, vy_std)

# Q-learning training loop
for episode in range(num_episodes):
    position = np.array([0, 0])  # Starting position (x, y)
    state = np.random.choice(num_states)  # Initialize hidden state from the HMM

    for t in range(seq_length):
        # Choose action based on epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = np.random.choice(num_actions)  # Explore
        else:
            action = np.argmax(Q_table[state])  # Exploit

        # Generate new emissions (velocities) based on the chosen state
        emission_x = np.random.normal(means[state, 0], std_devs[state, 0])
        emission_y = np.random.normal(means[state, 1], std_devs[state, 1])
        emission = np.array([emission_x, emission_y])

        # Update the agent's position based on the new emission (velocity)
        position = update_position(position, emission)

        # Compute the reward based on the new position
        reward = compute_reward(position, target)

        # Update the Q-table using the Bellman equation
        next_state = action  # Action corresponds to the next state
        Q_table[state, action] += alpha * (reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action])

        # Transition to the next state
        state = next_state

# Output the trained Q-table
print("Trained Q-table:")
print(Q_table)

# Simulate the navigation using the learned Q-table
position_history = []
position = np.array([0, 0])  # Starting position (x, y)

# Generate and store positions over time
for t in range(seq_length):
    state = np.random.choice(num_states)  # Initialize hidden state from the HMM

    # Choose action based on the learned Q-table
    action = np.argmax(Q_table[state])  # Exploit learned Q-values

    # Generate new emissions (velocities) from the AR-HMM based on the action
    emission_x = np.random.normal(means[action, 0], std_devs[action, 0])
    emission_y = np.random.normal(means[action, 1], std_devs[action, 1])
    emission = np.array([emission_x, emission_y])

    # Update the agent's position based on the new emission (velocity)
    position = update_position(position, emission)
    position_history.append(position)

# Convert position history to numpy array for easier indexing
position_history = np.array(position_history)

# Plot the 2D track of navigation colored by state
plt.figure(figsize=(10, 6))
for i in range(num_states):
    plt.scatter(position_history[i::num_states, 0], position_history[i::num_states, 1], label=f'State {i + 1}', alpha=0.6)

plt.scatter(target[0], target[1], color='red', marker='x', label='Target', s=100)  # Mark the target
plt.title("2D Track of Navigation Colored by States")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# Define the environment (target point in 2D space)
target = np.array([10, 10])  # Target coordinates (x, y)

# Define the number of states and actions
num_states = 3  # Hidden states from the AR-HMM
num_actions = 4  # Actions could be moving to neighboring hidden states

# Function to compute reward based on distance to target
def compute_reward(position, target):
    distance = np.linalg.norm(position - target)
    return -distance  # Negative reward for being far from the target

# Function to update the agent's position based on AR-HMM emissions (velocity)
def update_position(position, emission):
    return position + emission

# Objective function to evaluate the performance of the transition matrix
def objective_function(transition_matrix, means, std_devs, weights, num_episodes, seq_length):
    total_reward = 0
    position = np.array([0, 0])  # Starting position (x, y)

    for episode in range(num_episodes):
        # Initialize hidden state from the HMM
        state = np.random.choice(num_states)

        for t in range(seq_length):
            # Generate new emissions (velocities) from the AR-HMM
            emission_x = np.random.normal(means[state, 0], std_devs[state, 0])
            emission_y = np.random.normal(means[state, 1], std_devs[state, 1])
            emission = np.array([emission_x, emission_y])

            # Update the agent's position based on the new emission (velocity)
            position = update_position(position, emission)

            # Compute the reward based on the new position
            reward = compute_reward(position, target)
            total_reward += reward

            # If the agent is close to the target, break the episode
            if np.linalg.norm(position - target) < 0.1:
                break

    return total_reward

# Gradient of the objective function with respect to transition matrix parameters
def compute_gradients(transition_matrix, means, std_devs, weights, num_episodes, seq_length, epsilon=1e-5):
    gradients = np.zeros_like(transition_matrix)
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            # Perturb the parameter
            original_value = transition_matrix[i, j]
            transition_matrix[i, j] += epsilon
            loss_plus = objective_function(transition_matrix, means, std_devs, weights, num_episodes, seq_length)

            transition_matrix[i, j] -= 2 * epsilon
            loss_minus = objective_function(transition_matrix, means, std_devs, weights, num_episodes, seq_length)

            # Compute the gradient
            gradients[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

            # Restore original value
            transition_matrix[i, j] = original_value
            
    return gradients

# Gradient descent to optimize transition matrix parameters with normalization
def optimize_transition_matrix(transition_matrix, means, std_devs, weights, num_episodes, seq_length, learning_rate=0.01, max_iterations=100):
    for iteration in range(max_iterations):
        gradients = compute_gradients(transition_matrix, means, std_devs, weights, num_episodes, seq_length)
        transition_matrix -= learning_rate * gradients  # Update parameters
        
        # Normalize the transition matrix so that each row sums to 1
        transition_matrix = np.clip(transition_matrix, 0, None)  # Ensure non-negative
        transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)  # Normalize
        
        # Optional: Print loss at intervals
        if iteration % 10 == 0:
            loss = objective_function(transition_matrix, means, std_devs, weights, num_episodes, seq_length)
            print(f"Iteration {iteration}, Loss: {loss}")

    return transition_matrix


# Define parameters for the AR-HMM (three states)
transition_matrix = np.array([[0.7, 0.2, 0.1],
                              [0.3, 0.5, 0.2],
                              [0.2, 0.3, 0.5]])

# Mean velocities for each state in 2D (vx, vy)
means = np.array([[1.0, 0.5],  # State 1 mean velocities (vx, vy)
                  [0.5, 1.0],  # State 2 mean velocities (vx, vy)
                  [0.0, 1.5]]) # State 3 mean velocities (vx, vy)

# Standard deviations for each state (separate for vx and vy)
std_devs = np.array([[0.2, 0.1],  # State 1 std deviations (vx_std, vy_std)
                     [0.3, 0.2],  # State 2 std deviations (vx_std, vy_std)
                     [0.1, 0.3]]) # State 3 std deviations (vx_std, vy_std)

# Autoregressive weights for each state (for vx and vy)
weights = np.array([[0.7, 0.5],  # State 1 weights (wx, wy)
                    [0.4, 0.6],  # State 2 weights (wx, wy)
                    [0.6, 0.4]]) # State 3 weights (wx, wy)

# Run gradient descent to optimize the transition matrix
num_episodes = 50  # Reduced for faster optimization
seq_length = 50
optimized_transition_matrix = optimize_transition_matrix(transition_matrix, means, std_devs, weights, num_episodes, seq_length)

# Output the optimized transition matrix
print("Optimized Transition Matrix:")
print(optimized_transition_matrix)

# Simulate the navigation with optimized transition matrix
position_history = []
position = np.array([0, 0])  # Starting position (x, y)

# Generate and store positions over time
for t in range(seq_length):
    # Initialize hidden state from the HMM
    state = np.random.choice(num_states)

    # Generate new emissions (velocities) from the AR-HMM
    emission_x = np.random.normal(means[state, 0], std_devs[state, 0])
    emission_y = np.random.normal(means[state, 1], std_devs[state, 1])
    emission = np.array([emission_x, emission_y])

    # Update the agent's position based on the new emission (velocity)
    position = update_position(position, emission)
    position_history.append(position)

# Convert position history to numpy array for easier indexing
position_history = np.array(position_history)

# Plot the 2D track of navigation colored by state
plt.figure(figsize=(10, 6))
for i in range(num_states):
    plt.scatter(position_history[i::num_states, 0], position_history[i::num_states, 1], label=f'State {i + 1}', alpha=0.6)

plt.scatter(target[0], target[1], color='red', marker='x', label='Target', s=100)  # Mark the target
plt.title("2D Track of Navigation Colored by States")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()
