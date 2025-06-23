# Lunar Lander with Proximal Policy Optimization (PPO)

![alt text](https://gymnasium.farama.org/_images/lunar_lander.gif)

## Overview of Reinforcement Learning (RL)

Reinforcement Learning (RL) is a branch of machine learning where an agent learns optimal behavior by interacting with an environment. The agent observes the environment's state, takes actions, and receives feedback in the form of rewards. Over time, the agent refines its policy to maximize cumulative rewards, making RL ideal for sequential decision-making tasks.

In this project, RL is applied to the **Lunar Lander** problem. The goal is to train an agent to safely land a spacecraft on a designated landing pad, balancing fuel efficiency with precision. The agent learns through trial and error, guided by a reward function that incentivizes accurate landings and penalizes unsafe behaviors.

---

## Libraries Used

### 1. **Gymnasium**

* Gymnasium provides the environment (`LunarLander-v3`) for simulating the Lunar Lander task.
* Features customizable parameters like gravity, wind, and turbulence, allowing for a dynamic and challenging simulation.
* Includes rendering capabilities to visualize the agent's actions.

### 2. **Stable-Baselines3**

* A library that offers pre-implemented, state-of-the-art RL algorithms.
* Used to implement and train the agent using the **Proximal Policy Optimization (PPO)** algorithm.

---

## Proximal Policy Optimization (PPO)

### What is PPO?

PPO is a policy-gradient method designed for RL. Unlike value-based methods (like DQN), which rely on estimating state-action values, PPO directly optimizes the agent's policy (a mapping from states to actions). It uses the following key concepts:

* **Clipped Surrogate Objective**: Limits policy updates to prevent drastic changes, ensuring stable learning.
* **Value Function Estimation**: Reduces variance in gradient updates by incorporating value predictions.
* **Mini-Batch Updates**: Optimizes the policy using small subsets of data, improving efficiency and stability.

### Why PPO?

PPO was chosen for this project because:

1. **Stability**: The clipping mechanism ensures incremental updates, avoiding instability in policy learning.
2. **Sample Efficiency**: PPO learns effectively from limited data, reducing training time.
3. **Simplicity**: It is relatively easy to implement and tune compared to alternatives like TRPO or A3C.

### Comparison with Other Algorithms

* **DQN**: Effective for discrete action spaces but struggles with large, continuous spaces.
* **A2C**: Parallelizes training but lacks PPO's stability enhancements.
* **TRPO**: Ensures stability but is computationally expensive compared to PPO.

---

## Program Flow

### 1. **Setting Up the Environment**

The `LunarLander-v3` environment is initialized with:

* **Discrete Action Space**: Fixed thrust levels for simplicity.
* **Custom Gravity and Wind Parameters**: Introduces variability for training robustness.

### 2. **Custom Reward Wrapper**

A wrapper (`PrecisionLandingWrapper`) modifies the reward function to encourage precise landings:

* Rewards landings closer to the center of the landing pad.
* Penalizes imprecise or unsafe landings.

### 3. **Training the Agent with PPO**

* The PPO algorithm is used to train the agent for 200,000 time steps.
* A neural network with a multi-layer perceptron (MLP) policy is employed to map states to actions.
* The model optimizes the policy through iterative updates using the clipped surrogate objective.

### 4. **Saving and Loading the Model**

* The trained model is saved as `ppo_lunar_lander`.
* It can be reloaded to evaluate the agent’s performance without retraining.

### 5. **Evaluating the Trained Model**

* The agent’s performance is tested in the environment with rendering enabled.
* Observations showcase the agent's learned behavior in landing the spacecraft efficiently and accurately.

---

## Key Takeaways

1. **RL with PPO**:

   * Demonstrates the effectiveness of policy-gradient methods in complex control problems.
   * PPO provides a stable and efficient framework for training RL agents.

2. **Environment Customization**:

   * Modifying the reward function significantly enhances learning outcomes.

3. **Stable-Baselines3**:

   * Simplifies the implementation and training of advanced RL algorithms.

---

## Instructions to Run

1. **Install Required Libraries**:

   ```bash
   pip install pygame swig gymnasium[box2d] stable-baselines3
   ```

2. **Train the Model**:

   * Run the script to train the agent. Adjust the `total_timesteps` parameter as needed.

3. **Evaluate the Model**:

   * Load the trained model and observe the agent’s performance in the rendered environment.

4. **Customize Parameters** (Optional):

   * Experiment with gravity, wind, and reward modifications for varied training scenarios.
