# Assigment 2
# Svetlana Andrusenko

import numpy as np
import matplotlib.pyplot as plt
import json
from mse_vanilla import mean_squared_error as vanilla_mse
from mse_numpy import mean_squared_error as numpy_mse
from sklearn.metrics import mean_squared_error as sk_mse
import timeit as it
import os
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#1. Fix the mse_scaling_2.py code presented in the slides. Make sure to pass the
#test mse_vanilla == mse_numpy == mse_ske and print the time taken for each approach.

observed = [2, 4, 6, 8]
predicted = [2.5, 3.5, 5.5, 7.5]

karg = {'observed': observed,'predicted': predicted}
test = []
factory = {'mse_vanilla' : vanilla_mse,
            'mse_numpy' : numpy_mse,
            'mse_sk' : sk_mse
            }
#print(sk_mse.__doc__)
for talker, worker in factory.items():
    if talker == 'mse_vanilla' and worker == 'mse_numpy':
        exec_time = it.timeit('{worker(**karg)}', globals=globals(), number=100) / 100
        mse = worker(**karg)
    else: # changed: sk_mse requires y_true and y_pred as positional arguments (as in __doc__), not **karg
        exec_time = it.timeit('{worker(observed, predicted)}', globals=globals(), number=100) / 100
        mse = worker(observed, predicted)
    print(f"Mean Squared Error, {talker} :", mse, f"Average execution time: {exec_time} seconds")
    test.append(mse)
# test
print(test)
assert(test[0] == test[1] == test[2])
print('Test successful')

#-------------------------------------------------------------------------------------------------
# 2. Make a function that makes data: 1d oscillatory function with and without noise.
# The forced oscillations with a force oscillating at frequency Ω, obtained from the equation:
# mx" + kx = F0 cos(Ωt),
# with initial conditions at equilibrium position: x(0) = 0, x'(0) = 0

def oscillatory_function(t, F0 = 10, m = 0.5, k = 80, Omega = 12):
    """
    Computes the displacement x(t) for a forced oscillatory system described by the equation:
        mx" + kx = F0 cos(Omega * t)
    Parameters:
       t: time
       F0: amplitude of the external force. Default is 10 N
       m: mass. Default is 0.5 kg.
       k: spring constant. Default is 80 N/m.
       Omega: frequency of the external force. Default is 12 rad/s.
    Returns:
        numpy.ndarray: displacement x values corresponding to input time values.
    """
    w = np.sqrt(k/m)
    x = F0/m * 1/(Omega**2 - w**2) * (np.cos(w*t) - np.cos(Omega*t))
    return x

def create_data(points, mean, sd):
    """
    Generates time-series data for an oscillatory function with and without noise.
    Parameters:
    points: number of points.
    mean: mean value of the gaussian noise.
    sd: standard deviation of the gaussian noise.
    Returns:
    tuple: (t, x_truth, x_noise), where:
    - t: time
    - x_truth: true displacement values.
    - x_noise: displacement values with added noise.
    """
    t = np.linspace(0, 3, points)
    x_truth = oscillatory_function(t)
    x_noise = x_truth + np.random.normal(mean, sd, points)
    print(f'Data generated: {points} points, time range: {t[0]} to {t[-1]}, noise mean: {mean}, noise std dev: {sd}')
    return t, x_truth, x_noise

t, x_truth, x_noise = create_data(500, 0, 0.15)

plt.plot(t, x_truth, '-b', label = 'truth')
plt.scatter(t, x_noise, c='r', label='Noisy data')
plt.grid(True)
plt.legend()
plt.xlabel('t, s')
plt.ylabel('x(t), m')
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
file_path = os.path.join(base_dir, "data", "Oscillatory function with and without noise")
plt.savefig(file_path)
plt.show()

meta_info = {
    "Title": "Forced Oscillations with Noise",
    "Author": "Svetlana Andrusenko",
    "Description": (
        "The forced oscillations with a force oscillating at frequency Omega, obtained from the equation: "
        "mx'' + kx = F0 cos(Omega*t), with initial conditions at the equilibrium position: x(0)=0, x'(0)=0. "
        "Parameters: F0=10, m=0.5, k=80, Omega=12. "
        "Noise is Gaussian with mean=0 and std=0.15. "
        "Number of points: 500"
    )
}
file_path_m = os.path.join(base_dir, "data", "metadata_assigment2.json")
with open(file_path_m, "w") as f:
    json.dump(meta_info, f, indent=4)

#--------------------------------------------------------------------------------------------
# 3. Use clustering (pick the method you prefer) to group data and print the variance
# as a function of the number of clusters.
# The most popular - k-means
def cluster_and_plot_variance(t, x_noise, max_clusters = 15):
    """
    K-means clustering and plots variance as a function of the number of clusters.
    Parameters:
    t: time.
    x_noise: noisy displacement values.
    max_clusters: maximum number of clusters to test.
    """
    X = np.column_stack((t, x_noise)) # feature matrix
    var = []
    number_clusters = range(1, max_clusters + 1)
    print("K-Means Clustering Method (KMeans from sklearn)")
    print(f"Parameters: max_clusters={max_clusters}")
    for k in number_clusters:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        var.append(kmeans.inertia_) # sum of squared distances to centroids

    plt.figure()
    plt.plot(number_clusters, var, '-o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Variance')
    plt.title('Variance as a function of number of clusters')
    plt.grid(True)
    file_path1 = os.path.join(base_dir, "data", "Variance as a function of number of clusters")
    plt.savefig(file_path1)
    plt.show()

# plot clusters with cluster centers for n_clusters = 7
def cluster_and_plot_data(t, x_noise, n_clusters = 7):
    """
    K-means clustering and plotting the data with clusters in different colors.
    Parameters:
    t: time.
    x_noise: noisy displacement values.
    max_clusters: maximum number of clusters to test.
    """
    X = np.column_stack((t, x_noise))
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X) # assign each data point to a cluster
    centroids = kmeans.cluster_centers_ # cluster centers

    plt.figure()
    plt.scatter(t, x_noise, c=labels, cmap='viridis', marker='o', label='Clustered data')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label='Cluster centers')
    plt.plot(t, oscillatory_function(t), '-b', label='True function')
    plt.xlabel('t, s')
    plt.ylabel('x(t), m')
    plt.title('Clustered Oscillatory Data')
    plt.grid(True)
    plt.legend()
    file_path2 = os.path.join(base_dir, "data", "K-means clustering")
    plt.savefig(file_path2)
    plt.show()

cluster_and_plot_variance(t, x_noise)
cluster_and_plot_data(t, x_noise)

#------------------------------------------------------------------------------------------------------
# 4. Use LR, NN and PINNS to make a regression of such data.
# 5. Plot the solution as a function of the number of executed iterations (NN and PINNs).
# 6. Plot the error in respect to the truth as a function of the executed iteration
# (LR, NN and PINNs).
# 7. Assume to not know the truth any longer, select a method to monitor the progress
# of the calculations and plot its outcome (LR, NN and PINNs).

# LR (for task 4 and 6)
def LR(t, x_noise, x_truth):
    """
    LR and the error in respect to the truth as a function of the executed iteration.
    """
    T = t.reshape(-1, 1)
    model = LinearRegression()
    model.fit(T, x_noise)  # fit the model to the noisy data
    predictions = model.predict(T)  # predict values

    ms_errors_truth = sk_mse(x_truth, predictions)
    ms_errors_notruth = sk_mse(x_noise, predictions)

    plt.figure()
    plt.scatter(t, x_noise, c='gray', label='Noisy data')
    plt.plot(t, predictions, '-r', label='Linear Regression')
    plt.xlabel('t, s')
    plt.ylabel('x(t), m')
    plt.title('LR on Oscillatory Data')
    plt.legend()
    plt.grid(True)
    file_path3 = os.path.join(base_dir, "data", "LR")
    plt.savefig(file_path3)
    plt.show()

    plt.figure()
    plt.plot(range(1, len(predictions) + 1), np.full(len(predictions), ms_errors_truth), '-b')
    plt.xlabel('iteration')
    plt.ylabel('MSE')
    plt.title('LR MSE vs iterations')
    plt.grid(True)
    file_path4 = os.path.join(base_dir, "data", "LR_mse")
    plt.savefig(file_path4)
    plt.show()

    plt.figure()
    plt.plot(range(1, len(predictions) + 1), np.full(len(predictions), ms_errors_notruth), '-b')
    plt.xlabel('iteration')
    plt.ylabel('Loss')
    plt.title('LR loss vs iterations (we do not know truth)')
    plt.grid(True)
    file_path14 = os.path.join(base_dir, "data", "LR_loss (we do not know truth)")
    plt.savefig(file_path14)
    plt.show()

# Obviously, LR will give bad result
# the error will not change no matter how the weights are updated
LR(t, x_noise, x_truth)
print('Task completed LR')

# Let's do polynomial regression and compute and plot root mean square vs polynomial degree
def polynomial_regression_and_rms(t, x_noise, x_truth, max_degree=20):
    """
    Performs polynomial regression up to a given degree, plots MSE as a function of degree,
    and plots the best-fitting polynomial regression.
    """
    T = t.reshape(-1, 1)
    ms_errors = []
    degrees = range(1, max_degree + 1) # degrees to test
    best_degree = 1
    best_ms = float('inf')
    best_model = None

    for degree in degrees:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(T, x_noise)  # fit polynomial regression
        predictions = model.predict(T) # predicted values
        ms_error = sk_mse(x_truth, predictions) # MSE
        ms_errors.append(ms_error)

        if ms_error < best_ms: # update best model
            best_ms = ms_error
            best_degree = degree
            best_model = model

    plt.figure()
    plt.plot(degrees, ms_errors, '-o')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('MSE')
    plt.title('MSE vs Polynomial Degree')
    plt.grid(True)
    file_path5 = os.path.join(base_dir, "data", "MSE vs Polynomial Degree")
    plt.savefig(file_path5)
    plt.show()

    if best_model:
        best_predictions = best_model.predict(T)
        plt.figure()
        plt.scatter(t, x_noise, c='gray', label='Noisy Data')
        plt.plot(t, best_predictions, '-r', label=f'Best Polynomial Fit (Degree {best_degree})')
        plt.xlabel('t, s')
        plt.ylabel('x(t), m')
        plt.title('Polynomial Regression')
        plt.legend()
        plt.grid(True)
        file_path6 = os.path.join(base_dir, "data", "Polynomial Regression")
        plt.savefig(file_path6)
        plt.show()

polynomial_regression_and_rms(t, x_noise, x_truth)
print('Task completed polynomial regression')

# NN (task 4,5,6,7)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

def NN(t, x_noise, x_truth, epochs=1200, epoch_checkpoints=[10, 50, 100, 500, 1200]):
    """
    Trains a neural network on noisy data, plots predictions at different epochs.
    """
    model = Sequential([
        Input(shape=(1,)),
        Dense(256, activation='relu'),  # first hidden layer
        Dense(256, activation='relu'),  # second hidden layer
        Dense(256, activation='relu'),  # third hidden layer
        Dense(1, activation='linear')  # output layer
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')  # compile model with MSE loss

    predictions_at_epochs = {}
    error_at_epochs = {}
    error_at_epochs_notruth = {}
    for epoch in epoch_checkpoints:
        train_model = model.fit(t, x_noise, epochs=epoch, verbose=0)  # train model on noisy data
        predictions_at_epochs[epoch] = model.predict(t)  # store predictions
        #error_at_epochs[epoch] = np.sqrt(np.mean((x_truth - predictions_at_epochs[epoch].flatten()) ** 2))
        #error_at_epochs_notruth[epoch] = np.sqrt(np.mean((x_noise - predictions_at_epochs[epoch].flatten()) ** 2))  # Error in respect to noisy data
        error_at_epochs[epoch] = sk_mse(x_truth, predictions_at_epochs[epoch].flatten()) # error in respect to truth
        error_at_epochs_notruth[epoch] = sk_mse(x_noise, predictions_at_epochs[epoch].flatten()) # error in respect to noisy data

    plt.figure(figsize=(8, 5))
    plt.scatter(t, x_noise, c='gray', label='Noisy data')
    colors = ['r', 'b', 'g', 'm', 'c']
    for i, epoch in enumerate(epoch_checkpoints):
        plt.plot(t, predictions_at_epochs[epoch], color=colors[i % len(colors)], label=f'NN predictions (epoch {epoch})')

    plt.xlabel('t, s')
    plt.ylabel('x(t), m')
    plt.title('NN approximation of Noisy x(t) at Different Epochs')
    plt.legend()
    plt.grid(True)
    file_path7 = os.path.join(base_dir, "data", "NN approximation of Noisy x(t) at different Epochs")
    plt.savefig(file_path7)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_checkpoints, [error_at_epochs[ep] for ep in epoch_checkpoints], '-*b')
    plt.xlabel('Epochs')
    plt.ylabel('Error (MSE)')
    plt.title('NN error over selected epochs')
    plt.grid(True)
    file_path8 = os.path.join(base_dir, "data", "NN error over selected epochs")
    plt.savefig(file_path8)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_checkpoints, [error_at_epochs_notruth[ep] for ep in epoch_checkpoints], '-*b')
    plt.xlabel('Epochs')
    plt.ylabel('Error in respect to Noisy Data (MSE)')
    plt.title('NN error over selected epochs (do not know truth)')
    plt.grid(True)
    file_path9 = os.path.join(base_dir, "data", "NN error over selected epochs (do not know truth)")
    plt.savefig(file_path9)
    plt.show()

    return model

NN(t, x_noise, x_truth)
print('Task completed NN')

# PINNs(task 4,5,6,7)
# The forced oscillations with a force oscillating at frequency Ω, obtained from the equation:
# mx" + kx = F0 cos(Ωt),
# with initial conditions at equilibrium position: x(0) = 0, x'(0) = 0

import torch
import torch.nn as nn
import torch.optim as optim

F0 = 10
m = 0.5
k = 80
Omega = 12

class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 256), SinActivation(),
            nn.Linear(256, 256), SinActivation(),
            nn.Linear(256, 256), SinActivation(),
            nn.Linear(256, 1)
        )

    def forward(self, t):
        return self.net(t)

def loss_physics_function(model, t):
    t.requires_grad = True
    x = model(t)
    dx_dt = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    d2x_dt2 = torch.autograd.grad(dx_dt, t, torch.ones_like(dx_dt), create_graph=True)[0]
    physics_loss = torch.mean((m * d2x_dt2 + k * x - F0 * torch.cos(Omega * t)) ** 2)
    return physics_loss

t0 = torch.tensor([[0.0]], requires_grad=True)
def initial_loss(model):
    x0 = torch.tensor([[0.0]], dtype=torch.float32)
    dx0 = torch.tensor([[0.0]], dtype=torch.float32)
    x_pred = model(t0)
    dx_pred = torch.autograd.grad(x_pred, t0, torch.ones_like(x_pred), create_graph=True)[0]
    loss_init = torch.mean((x_pred - x0)**2 + (dx_pred - dx0)**2)
    return loss_init

model = PINN()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

t_train = torch.linspace(0, 2, 500).view(-1, 1)
t_train.requires_grad = True

num_epochs = 12000
save_epochs = [10, 1000, 5000, 7000, num_epochs - 1]
saved_predictions = {}
loss_values = []
mse_errors = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    # Multiplying initial_loss by 30 because it had a very small weight
    # compared to the physics-based loss, which prevented proper enforcement of initial conditions.
    loss = loss_physics_function(model, t_train) + 30 * initial_loss(model)
    loss.backward()
    optimizer.step()
    loss_values.append(loss.item())

    if epoch in save_epochs:
        t_test = torch.linspace(0, 2, 500).view(-1, 1)
        saved_predictions[epoch] = model(t_test).detach().numpy()
    # if epoch % 500 == 0:
    #     print(f'Epoch {epoch}, Loss: {loss.item():.6f}')

t_test = torch.linspace(0, 2, 500).view(-1, 1)
x_exact = oscillatory_function(t_test.numpy())

plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i, epoch in enumerate(save_epochs):
    plt.plot(t_test.numpy(), saved_predictions[epoch], label=f'PINN (epoch {epoch})', color=colors[i])

plt.plot(t_test.numpy(), x_exact, '--', label='Exact Solution', color='black', linewidth=2)
plt.legend()
plt.xlabel('t, s')
plt.ylabel('x(t), m')
plt.title('Forced oscillations solved with PINN')
plt.grid(True)
file_path10 = os.path.join(base_dir, "data", "Forced oscillations solved with PINN")
plt.savefig(file_path10)
plt.show()

for epoch in save_epochs:
    x_pinn = saved_predictions[epoch]
    mse = sk_mse(x_exact, x_pinn)
    mse_errors.append(mse)

plt.figure(figsize=(8, 5))
plt.plot(save_epochs, mse_errors, marker='o', linestyle='-', color='blue', label="MSE")
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('PINN error over selected epochs')
plt.grid(True)
plt.legend()
file_path11 = os.path.join(base_dir, "data", "PINN error")
plt.savefig(file_path11)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(save_epochs, [loss_values[e] for e in save_epochs], marker='o', linestyle='-', label="Loss at selected epochs")
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('PINN loss over selected epochs (we do not know truth)')
plt.grid(True)
plt.legend()
file_path12 = os.path.join(base_dir, "data", "PINN loss (we do not know truth)")
plt.savefig(file_path12)
plt.show()

print('Task completed PINN')

# 8. Run the reinforcement learning script.
# Plot the number of iterations needed to converge as a function of the learning rate.
# This is the copy of reinforcement learning script:
import matplotlib.animation as animation
# GridWorld Environment
class GridWorld:
    """GridWorld environment with obstacles and a goal.
    The agent starts at the top-left corner and has to reach the bottom-right corner.
    The agent receives a reward of -1 at each step, a reward of -0.01 at each step in an obstacle, and a reward of 1 at the goal.

    Args:
        size (int): The size of the grid.
        num_obstacles (int): The number of obstacles in the grid.

    Attributes:
        size (int): The size of the grid.
        num_obstacles (int): The number of obstacles in the grid.
        obstacles (list): The list of obstacles in the grid.
        state_space (numpy.ndarray): The state space of the grid.
        state (tuple): The current state of the agent.
        goal (tuple): The goal state of the agent.

    Methods:
        generate_obstacles: Generate the obstacles in the grid.
        step: Take a step in the environment.
        reset: Reset the environment.
    """
    def __init__(self, size=5, num_obstacles=5):
        self.size = size
        self.num_obstacles = num_obstacles
        self.obstacles = [(0, 4), (4, 3), (1, 3), (1, 0), (3, 2)]
        self.state_space = np.zeros((self.size, self.size))
        self.state = (0, 0)
        self.goal = (self.size-1, self.size-1)

    def step(self, action):
        """
        Take a step in the environment.
        The agent takes a step in the environment based on the action it chooses.

        Args:
            action (int): The action the agent takes.
                0: up
                1: right
                2: down
                3: left

        Returns:
            state (tuple): The new state of the agent.
            reward (float): The reward the agent receives.
            done (bool): Whether the episode is done or not.
        """
        x, y = self.state

        if action == 0:  # up
            x = max(0, x-1)
        elif action == 1:  # right
            y = min(self.size-1, y+1)
        elif action == 2:  # down
            x = min(self.size-1, x+1)
        elif action == 3:  # left
            y = max(0, y-1)
        self.state = (x, y)
        if self.state in self.obstacles:
         #   self.state = (0, 0)
            return self.state, -1, True
        if self.state == self.goal:
            return self.state, 1, True
        return self.state, -0.1, False

    def reset(self):
        """
        Reset the environment.
        The agent is placed back at the top-left corner of the grid.

        Args:
            None

        Returns:
            state (tuple): The new state of the agent.
        """
        self.state = (0, 0)
        return self.state


# Q-Learning
class QLearning:
    """
    Q-Learning agent for the GridWorld environment.

    Args:
        env (GridWorld): The GridWorld environment.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.
        episodes (int): The number of episodes to train the agent.

    Attributes:
        env (GridWorld): The GridWorld environment.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.
        episodes (int): The number of episodes to train the agent.
        q_table (numpy.ndarray): The Q-table for the agent.

    Methods:
        choose_action: Choose an action for the agent to take.
        update_q_table: Update the Q-table based on the agent's experience.
        train: Train the agent in the environment.
        save_q_table: Save the Q-table to a file.
        load_q_table: Load the Q-table from a file.
    """
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=0.1, episodes=10):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = np.zeros((self.env.size, self.env.size, 4))

    def choose_action(self, state):
        """
        Choose an action for the agent to take.
        The agent chooses an action based on the epsilon-greedy policy.

        Args:
            state (tuple): The current state of the agent.

        Returns:
            action (int): The action the agent takes.
                0: up
                1: right
                2: down
                3: left
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice([0, 1, 2, 3])  # exploration
        else:
            return np.argmax(self.q_table[state])  # exploitation

    def update_q_table(self, state, action, reward, new_state):
        """
        Update the Q-table based on the agent's experience.
        The Q-table is updated based on the Q-learning update rule.

        Args:
            state (tuple): The current state of the agent.
            action (int): The action the agent takes.
            reward (float): The reward the agent receives.
            new_state (tuple): The new state of the agent.

        Returns:
            None
        """
        self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + \
            self.alpha * (reward + self.gamma * np.max(self.q_table[new_state]))

    def train(self):
        """
        Train the agent in the environment.
        The agent is trained in the environment for a number of episodes.
        The agent's experience is stored and returned.

        Args:
            None

        Returns:
            rewards (list): The rewards the agent receives at each step.
            states (list): The states the agent visits at each step.
            starts (list): The start of each new episode.
            steps_per_episode (list): The number of steps the agent takes in each episode.
        """
        rewards = []
        states = []  # Store states at each step
        starts = []  # Store the start of each new episode
        steps_per_episode = []  # Store the number of steps per episode
        steps = 0  # Initialize the step counter outside the episode loop
        episode = 0
        #print(self.q_table)
        while episode < self.episodes:
            state = self.env.reset()
            total_reward = 0
            done = False
            #print(f"Episode {episode+1}")
            #print(self.q_table)
            while not done:
                action = self.choose_action(state)
                new_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, new_state)
                state = new_state
                total_reward += reward
                states.append(state)  # Store state
                steps += 1  # Increment the step counter
                if done and state == self.env.goal:  # Check if the agent has reached the goal
                    starts.append(len(states))  # Store the start of the new episode
                    rewards.append(total_reward)
                    steps_per_episode.append(steps)  # Store the number of steps for this episode
                    steps = 0  # Reset the step counter
                    episode += 1
        return rewards, states, starts, steps_per_episode


for i in range(1):
    env = GridWorld(size=5, num_obstacles=5)
    agent = QLearning(env)

    # Train the agent and get rewards
    rewards, states, starts, steps_per_episode = agent.train()  # Get starts and steps_per_episode as well

    # Visualize the agent moving in the grid
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(i):
        """
        Update the grid with the agent's movement.

        Args:
            i (int): The current step.

        Returns:
            None
        """
        ax.clear()
        # Calculate the cumulative reward up to the current step
        #print(rewards)
        cumulative_reward = sum(rewards[:i+1])
        #print(rewards[:i+1])
        # Find the current episode
        current_episode = next((j for j, start in enumerate(starts) if start > i), len(starts)) - 1
        # Calculate the number of steps since the start of the current episode
        if current_episode < 0:
            steps = i + 1
        else:
            steps = i - starts[current_episode] + 1
        ax.set_title(f"Episode: {current_episode+1}, Number of Steps to Reach Target: {steps}")
        grid = np.zeros((env.size, env.size))
        for obstacle in env.obstacles:

            grid[obstacle] = -1
        grid[env.goal] = 1
        grid[states[i]] = 0.5  # Use states[i] instead of env.state
        ax.imshow(grid, cmap='magma')

    ani = animation.FuncAnimation(fig, update, frames=range(len(states)), repeat=False)
    print(f"Environment number {i+1}")
    for i, steps in enumerate(steps_per_episode, 1):
        print(f"Episode {i}: {steps} Number of Steps to Reach Target ")
    #print(f"Total reward: {sum(rewards):.2f}")
    print()
    #ani.save('gridworld_lin.gif', writer='pillow', dpi=100)

    plt.show()
    #plt.close()

#Plot the number of iterations needed to converge as a function of the learning rate.
alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
iterations_needed = []
required_goal_streak = 3 #means requires 3 episodes in a row in which the agent reaches the goal
max_episodes = 2000

for alpha in alphas:
    env = GridWorld(size=5)
    agent = QLearning(env, alpha=alpha, gamma=0.95, epsilon=0.1)
    success_streak = 0
    total_steps = 0
    episode_count = 0

    while episode_count < max_episodes:
        state = env.reset()
        done = False
        steps_this_episode = 0
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            new_state, reward, done = env.step(action)
            agent.update_q_table(state, action, reward, new_state)
            state = new_state
            total_reward += reward
            steps_this_episode += 1
            # if the episode lasts too long (more than 500 steps), interrupt it
            if steps_this_episode > 500:
                done = True

        total_steps += steps_this_episode
        episode_count += 1

        if total_reward > 0:
            success_streak += 1
        else:
            success_streak = 0

        if success_streak >= required_goal_streak:
            break

    iterations_needed.append(total_steps)
    print(f"Alpha={alpha}, steps to converge={total_steps}, episodes used={episode_count}")

plt.figure(figsize=(7, 5))
plt.plot(alphas, iterations_needed, '-o')
plt.xlabel('Learning rate')
plt.ylabel('Iterations to converge')
plt.title('Iterations needed to converge as a function of the learning rate')
plt.grid(True)
file_path20 = os.path.join(base_dir, "data", "Iterations needed to converge as a function of the learning rate (RL)")
plt.savefig(file_path20)
plt.show()