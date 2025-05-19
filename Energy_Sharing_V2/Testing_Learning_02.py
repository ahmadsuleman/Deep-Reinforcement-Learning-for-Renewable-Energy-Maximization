# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 15:24:25 2021

@author: Suleman_Sahib
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 13:54:31 2021

@author: Suleman_Sahib
"""

#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tensorflow as tf
import gym
from gym.wrappers import Monitor
from collections import deque
import tensorflow as tf
import numpy as np
import random
import math
import time
import glob
import io
import base64
from Reward_function_MDP import ship_movement
import matplotlib.pyplot as plt


# In[11]:

#env = gym.make('CartPole-v0')
env = ship_movement()
state = env.reset()
num_features = state.shape[0]
num_actions = env.action_space.n
print('Number of state features: {}'.format(num_features))
print('Number of possible actions: {}'.format(num_actions))

class DQN(tf.keras.Model):
  """Dense neural network class."""
  def __init__(self):
    super(DQN, self).__init__()
    self.dense1 = tf.keras.layers.Dense(32, activation="relu")
    self.dense2 = tf.keras.layers.Dense(32, activation="relu")
    self.dense3 = tf.keras.layers.Dense(32, activation="relu")
    self.dense4 = tf.keras.layers.Dense(32, activation="relu")
    self.dense5 = tf.keras.layers.Dense(1, activation="relu")
    self.dense6 = tf.keras.layers.Dense(num_actions, activation="softmax") # No activation
    
  def call(self, x):
    """Forward pass."""
    x = self.dense1(x)
    x = self.dense2(x)
    x1 = self.dense3(x)
    x2 = self.dense4(x)
    critic = self.dense5(x1)
    actor = self.dense6(x2)
    #q_value = value + (advantage - tf.math.reduce_mean(advantage))
    
    return actor, critic

main_nn = DQN()
target_nn = DQN()

#optimizer = tf.keras.optimizers.Adam(1e-5)
mse = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
main_nn.build(input_shape=(None,num_features))
main_nn.load_weights("TF_MODEL_AC.h5")
huber_loss = tf.keras.losses.Huber()

# Configuration parameters for the whole setup
gamma = 0.98  # Discount factor for past rewards
max_steps_per_episode = 24
eps = np.finfo(np.float32).eps.item() 

action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0
loss_a = []
reward_hi = []
rew = 0
last_100_ep_rewards = []

for ti in range(70000):  # Run until solved
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = main_nn(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            state, reward, done = env.step(action)
            #env.render_grid()
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            diff = ret - value
            actor_losses.append(-log_prob * diff)
            
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, main_nn.trainable_variables)
        optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()
        
        # Log details
    episode_count += 1
    if len(last_100_ep_rewards) == 100:
       last_100_ep_rewards = last_100_ep_rewards[1:]
    last_100_ep_rewards.append(running_reward) 
    
    if episode_count % 50 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))
        loss_a.append(loss_value)
        reward_hi.append(running_reward)
        
    if running_reward > 950:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        main_nn.save_weights("TF_MODEL_AC.h5")
        #
        rew = running_reward
        print("Weights Saved")
        #break
plt.plot(loss_a)
plt.show()
plt.plot(reward_hi)
plt.show()
np.save("Reward_AC.npy",reward_hi,allow_pickle=True)
np.save("Loss_AC.npy",loss_a,allow_pickle=True)