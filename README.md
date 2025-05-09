# Renewable Energy Maximization for Pelagic Islands Microgrids Using Deep Reinforcement Learning

This repository contains the full implementation of the project proposed in our research article:

**"Renewable Energy Maximization for Pelagic Islands Network of Microgrids Through Battery Swapping Using Deep Reinforcement Learning"**  
Published in *IEEE Access*, 2023.  
DOI: [10.1109/ACCESS.2023.3302895](https://doi.org/10.1109/ACCESS.2023.3302895)

---

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

## 📂 Repository Structure

```bash
RE-Maximization-PINMG-DRL/
│
├── README.md                  # This file
├── LICENSE
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment (optional)
├── .gitignore
│
├── data/                      # Input datasets (synthetic or real)
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── models/                    # Trained DQN models and architectures
│   ├── trained_models/
│   └── model_architectures/
│
├── src/
│   ├── environment/           # Island microgrid simulation
│   ├── agents/                # DQN, Dueling DQN agents
│   ├── scheduling/            # Two-stage cooperative scheduling logic
│   ├── utils/                 # Logging, visualization, helpers
│   └── main.py                # Entry point to run the full simulation
│
├── notebooks/
│   └── analysis.ipynb         # Training curves, RE usage plots, ablation
│
└── results/
    ├── figures/               # Graphs, bar plots, comparisons
    └── logs/                  # Experiment logs
