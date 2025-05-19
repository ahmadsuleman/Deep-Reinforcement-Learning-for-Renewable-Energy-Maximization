# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 22:29:39 2021

@author: Suleman_Sahib
"""



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
    self.dense6 = tf.keras.layers.Dense(num_actions, dtype=tf.float32) # No activation
    
  def call(self, x):
    """Forward pass."""
    x = self.dense1(x)
    x = self.dense2(x)
    x1 = self.dense3(x)
    x2 = self.dense4(x)
    value = self.dense5(x1)
    advantage = self.dense6(x2)
    q_value = value + (advantage - tf.math.reduce_mean(advantage))
    
    return q_value

main_nn = DQN()
target_nn = DQN()


main_nn.build(input_shape=(None,num_features))
main_nn.load_weights("Duling_and_DDQN_TF_MODEL.h5")

optimizer = tf.keras.optimizers.Adam(1e-4)
mse = tf.keras.losses.MeanSquaredError()


# In[10]:




# In[12]:


class ReplayBuffer(object):
  """Experience replay buffer that samples uniformly."""
  def __init__(self, size):
    self.buffer = deque(maxlen=size)

  def add(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def __len__(self):
    return len(self.buffer)

  def sample(self, num_samples):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    idx = np.random.choice(len(self.buffer), num_samples)
    for i in idx:
      elem = self.buffer[i]
      state, action, reward, next_state, done = elem
      states.append(np.array(state, copy=False))
      actions.append(np.array(action, copy=False))
      rewards.append(reward)
      next_states.append(np.array(next_state, copy=False))
      dones.append(done)
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions)
    rewards = np.array(rewards, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)
    dones = np.array(dones, dtype=np.float32)
    return states, actions, rewards, next_states, dones


# In[5]:


def select_epsilon_greedy_action(state, epsilon):
  """Take random action with probability epsilon, else take best action."""
  result = tf.random.uniform((1,))
  if result < epsilon:
    return env.action_space.sample() # Random action (left or right).
  else:
    return tf.argmax(main_nn(state)[0]).numpy() # Greedy action for state.

@tf.function
def train_step(states, actions, rewards, next_states, dones):
    next_qs = target_nn(next_states)
    act_value = main_nn(next_states)
    max_index = tf.math.argmax(act_value, axis=-1)
    max_index= tf.reshape(max_index, (max_index.shape[0],1))
    max_next_qs = tf.gather_nd(next_qs, max_index, batch_dims=1, name=None)
    
    target = rewards + (1. - dones) * discount * max_next_qs
    with tf.GradientTape() as tape:
        qs = main_nn(states)
        action_masks = tf.one_hot(actions, num_actions)
        masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
        loss = mse(target, masked_qs)
    grads = tape.gradient(loss, main_nn.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
    return loss


# In[13]:


espi = []
# Hyperparameters.
num_episodes = 5000
epsilon = 1.0
start = 1.0
end = 0.0001
decay1 = 0.01
#
# 1000 episodes
epsilon_decay = 0.0019
# 5000
#epsilon_decay_1 = 0.0002

#10,000 Episodes Single
#epsilon_decay_1 = 0.000095
#10,000 Episodes Double
#epsilon_decay_1 = 0.000006

#epsilon_decay_2 = 0.00046

#25000 Double 
#epsilon_decay_1 = 0.000006

#epsilon_decay_2 = 0.000176

# 70 K double
epsilon_decay_1 = 0.000005
epsilon_decay_2 = 0.0000372



batch_size = 32
discount = 0.98
buffer = ReplayBuffer(100000)
cur_frame = 0
min_reward = 980


# Start training. Play game once and then train with a batch.
last_100_ep_rewards = []
loss_value = []
loss_value_1 = []
rew = []
ep_reward = 0
for episode in range(num_episodes+1):
  state = env.reset()
  #if ep_reward > min_reward:
        #main_nn.save_weights("TF_MODEL_1.h5")
                # serialize model to JSON
        
        # serialize weights to HDF5
        #main_nn.save_weights("TF_MODEL_2.h5")
        #print("Saved model to disk")
        #min_reward = ep_reward
  
  ep_reward, done = 0, False
  loss = 0
  while not done:
    state_in = tf.expand_dims(state, axis=0)
    action = select_epsilon_greedy_action(state_in, epsilon)
    next_state, reward, done = env.step(action)
    ep_reward += reward
    # Save to experience replay.
    buffer.add(state, action, reward, next_state, done)
    state = next_state
    cur_frame += 1
    # Copy main_nn weights to target_nn.
    if cur_frame % 500 == 0:
      target_nn.set_weights(main_nn.get_weights())

    # Train neural network.
    if len(buffer) >= batch_size:
      states, actions, rewards, next_states, dones = buffer.sample(batch_size)
      loss = train_step(states, actions, rewards, next_states, dones)
      
      
  
  
  epsilon = end + (start - end) * math.exp(-1. * episode * decay1)
        
  #if (episode > 50000) and (episode < 70000):
  #      epsilon -= epsilon_decay_2  
    
  
      

  if len(last_100_ep_rewards) == 100:
    last_100_ep_rewards = last_100_ep_rewards[1:]
  last_100_ep_rewards.append(ep_reward) 
  if (epsilon < 0.2) and (np.mean(last_100_ep_rewards) > min_reward):
      main_nn.save_weights("Duling_and_DDQN_TF_MODEL_1.h5")
      min_reward = np.mean(last_100_ep_rewards)
      print("Weights Saved")
  
  
  if episode % 1 == 0:
    print(f'Episode {episode}/{num_episodes}. Epsilon: {epsilon:.3f}. '
          f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.3f}')
    loss_value.append(loss)
    rew.append(np.mean(last_100_ep_rewards))
    espi.append(epsilon)
        
        
    

plt.plot(loss_value)
plt.show()



plt.plot(rew)
plt.show()
np.save("LOSS_VALUE_Duling_and_DDQN_2.npy", loss_value, allow_pickle=True)
np.save("Reward_Duling_and_DDQN_2.npy", rew, allow_pickle=True)
np.save("Epsilon_Duling_and_DDQN_2.npy", espi, allow_pickle=True) 



# In[ ]:




