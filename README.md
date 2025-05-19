# Renewable Energy Maximization for Pelagic Islands Microgrids Using Deep Reinforcement Learning

This repository contains the full implementation of the project proposed in our research article:

**"Renewable Energy Maximization for Pelagic Islands Network of Microgrids Through Battery Swapping Using Deep Reinforcement Learning"** 
** Abstract **
The study proposes an energy management system of pelagic islands network microgrids
(PINMGs) based on reinforcement learning (RL) under the effect of environmental factors. Furthermore, the
day-ahead standard scheduling proposes an energy-sharing framework across islands by presenting a novel
method to optimize the use of renewable energy (RE). Energy sharing across islands is critical for powering
isolated islands that need electricity owing to a lack of renewable energy supplies to fulfill local demand.
A two-stage cooperative multi-agent deep RL solution based on deep Q-learning (DQN) with central RL
and island agents (IA) spread over several islands has been presented to tackle this difficulty. Because of its
in-depth learning potential, deep RL-based systems effectively train and optimize their behaviors across
several epochs compared to other machine learning or traditional methods. As a result, the centralized
RL-based problem of scheduling charge battery sharing from resource-rich islands (SI) to load island
networks (LIN) was addressed utilizing dueling DQN. Furthermore, due to its precise tracking, the case
study compared the accuracy of various DQN approaches and further scheduling based on the dueling DQN.
The need for LIN is also stochastic because of variable demand and charging patterns. Hence, the simulation
results, including energy scheduling through the ship, are confirmed by optimizing RE consumption via
sharing across several islands, and the effectiveness of the proposed method is validated by state and action
perturbation to guarantee robustness.

Published in *IEEE Access*, 2023.  
DOI: [10.1109/ACCESS.2023.3302895](https://doi.org/10.1109/ACCESS.2023.3302895)

---
# Project Overview

<div style="display: flex; justify-content: space-around;">
  <img src="/Energy_Sharing_V2/Results/Images/overview.gif" alt="Sharing Strcture" width="500" />
  <img src="/Energy_Sharing_V2/Results/Images/overview2.gif" alt="Battery Swaping" width="500" />
  <img src="/Energy_Sharing_V2/Results/Images/overview1.gif" alt="Island Structure" width="500" />
  <img src="/Energy_Sharing_V2/Results/Images/1_Functionality_Profile_1000.png" alt="Island Structure" width="500" />
</div>

## 📘 Project Scope

### 🌍 Background
Pelagic islands—isolated islands far from mainland grids—rely heavily on local renewable energy (RE) sources. However, due to unpredictable solar/wind conditions, some islands face shortages while others may have surplus energy. Establishing an effective and intelligent energy-sharing framework between these islands is critical to ensure consistent and optimal power supply.

### 🎯 Objective
The objective of this project is to **maximize the utilization of renewable energy** across a network of isolated microgrids (PINMGs) by enabling **cooperative energy sharing** via **battery swapping using ships**.

---

## 🧠 Technical Approach

### 🔁 Two-Stage DRL Architecture
We propose a **two-stage cooperative Multi-Agent Deep Reinforcement Learning (MADRL)** approach that consists of:

1. **Central Reinforcement Learning (CRL)**  
   - Manages global-level scheduling decisions.
   - Determines optimal energy transfer from Source Islands (SIs) to Load Island Networks (LINs).

2. **Island Agents (IAs)**  
   - Operate locally on each island.
   - Handle individual energy demand/supply dynamics using local policies.

### 🧮 Algorithms Used
- **Dueling Deep Q-Networks (Dueling DQN)** for improved convergence and policy evaluation.
- **Environment Modeling** includes stochastic demand, RE generation, and dynamic battery shipping delays.
- **State and Action Perturbation Tests** to validate robustness and generalizability.

### 🛳️ Battery Swapping Mechanism
Ships transport charged batteries from surplus-producing islands to demand-heavy islands. The cost and schedule of transport are factored into the optimization.

---
# Code Explaination 
---
## Script: *Reward_function_MDP.py*
### 📌 Features

- 12 States discrete environment with islands acting as microgrids present at different states initially unknown to the agent.
- Three island types:
  - **Source Island (SI)**: Exports batteries.
  - **Source Load Island (SLI)**: Exports batteries, also has significant loads.
  - **Load Island (LIN)**: Only consumes batteries.
- Ship movement across grid cells with discrete actions.
- Battery collection and delivery logic.
- Time-series simulation of energy generation and consumption.
- Reinforcement Learning-ready environment with step and reset functions.
- TensorFlow-based DQN setup included as a test section.

### 🧠 Objectives

The goal is to train an agent (ship) to:

- Efficiently transport batteries from source islands to the load island.
- Maximize delivery success within step limits.
- Minimize energy from non-renewable sources (penalized in rewards).

This helps separate the state value from the relative advantages of each action.

---

## 🕹️ Environment

The `ship_movement` environment is implemented in the file `Reward_function_MDP.py`. It defines:

- `reset()`: Initializes the environment.
- `step(action)`: Applies an action and returns (next_state, reward, done).
- `action_space`: A Gym-compatible discrete action space.
- `state`: An array of features representing the ship’s status.
---
## 🚀 Dueling DQN for Ship Navigation Using TensorFlow
## Script: *TF_Dueling_DQN.py*
This project implements a **Dueling Deep Q-Network (DQN)** in TensorFlow to train an agent in a custom OpenAI Gym environment called `ship_movement`. The agent learns to optimize the battery pickup & delivery strategy of a simulated ship using reinforcement learning principles.

---

## ⚙️ Hyperparameters

| Parameter         | Value          |
|------------------|----------------|
| Episodes         | 5000           |
| Batch Size       | 32             |
| Discount Factor  | 0.98           |
| Learning Rate    | 0.0001         |
| Epsilon (start)  | 1.0            |
| Epsilon (end)    | 0.0001         |
| Epsilon Decay    | Exponential    |
| Replay Buffer    | 100,000 steps  |
| Target Sync Freq | 500 steps      |

---

## 🔁 Training Loop

Each episode includes the following steps:

1. Reset the environment and initialize variables.
2. Choose actions using the epsilon-greedy policy.
3. Store experiences in the replay buffer.
4. Sample random mini-batches to train the model.
5. Periodically update the target network.
6. Reduce epsilon over time to shift from exploration to exploitation.
7. Save model weights if performance improves.

---

# 🚢 Deep Reinforcement Learning for Ship Navigation

This repository contains implementations of various deep reinforcement learning (DRL) algorithms applied to a custom ship navigation environment. The goal is to train agents (using DQN, Double DQN, Dueling DQN, Actor-Critic) to navigate efficiently, possibly avoiding islands or minimizing cost over routes.

---

## 📁 Project Structure

### 🧠 RL Training Scripts

| File | Description |
|------|-------------|
| `Dueling_DQN_with DDQN.py` | Main training script implementing **Dueling DQN** with **Double DQN** extensions using TensorFlow. |
| `TF_DQN_01.py` | TensorFlow-based implementation of the standard **DQN** algorithm. |
| `TF_DDQN_01.py` | Implements **Double DQN** to reduce overestimation bias. |
| `TF_Duling_DQN.py` | Likely alternative or enhanced version of the Dueling DQN script. |
| `tf_actor_critic.py` | TensorFlow implementation of an **Actor-Critic** reinforcement learning agent. |

---

### 🛠️ Environment and Utilities

| File | Description |
|------|-------------|
| `Reward_function_MDP.py` | Defines the `ship_movement` class, a custom Gym environment for training. |
| `Island_Generator.py` | Generates island map configurations or obstacles for the environment. |
| `Modules.py` | Contains helper functions and reusable logic modules. |
| `decay_settings.py` | Stores epsilon decay strategies and other hyperparameters. |
| `price_dataset_file.py` | Loads or simulates pricing/cost data, likely used for energy optimization tasks. |

---

### 📊 Visualization & Plotting

| File | Description |
|------|-------------|
| `Functionality Plot.py` | Visualizes model performance, reward trends, or Q-values. |
| `Ploting_and_Saving.py` | Automates plotting and saving of training metrics like reward, loss, and epsilon. |
| `Plot_RL_results.py` | Standalone script for plotting saved `.npy` result files. |

---

### 🧪 Testing & Evaluation

| File | Description |
|------|-------------|
| `Testing_Learning.py` | Tests trained model performance in the custom environment. |
| `Testing_learning_01.py` | Variant testing script with a different configuration or episode limit. |
| `Testing_Learning_02.py` | Another evaluation scenario or experiment. |

---

## 🗃️ Datasets and Scenario Files

| Name | Description |
|------|-------------|
| `datasets/` | Folder for trajectory, pricing, or environment data used during training. |
| `straight/` | Contains configurations for straight-line navigation environments. |
| `Reward_RL_300_straight.rar` | Compressed file with reward data for 300 episodes in the straight scenario. |
| `straight.rar` | Compressed version of the `straight/` directory. |

---

## 💾 Model Artifacts & Outputs

| Name | Description |
|------|-------------|
| `RandomModel/` | Stores random or baseline model variants. |
| `Results/` | Output directory for saved weights, plots, and reward/loss history. |
| `__pycache__/` | Auto-generated folder with cached Python bytecode (`.pyc` files). |

---

## ✅ How to Get Started

- **Install requirements**:
   ```bash
   
   git clone https://github.com/eagle-Ji/Deep-Reinforcement-Learning-for-Renewable-Energy-Maximization
   cd  Deep-Reinforcement-Learning-for-Renewable-Energy-Maximization
   pip install -r requirements.txt
   ```
   
## References

- Amin, M. A., Suleman, A., Waseem, M., Iqbal, T., Aziz, S., Faiz, M. T., ... & Saleh, A. M. (2023). Renewable energy maximization for pelagic islands network of microgrids through battery swapping using deep reinforcement learning. *IEEE Access*, 11, 86196-86213. [DOI: 10.1109/ACCESS.2023.3302895](https://doi.org/10.1109/ACCESS.2023.3302895)

   


