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
#main_nn.load_weights("Duling_and_DDQN_TF_MODEL.h5")
main_nn.load_weights("Duling_DQN_TF_MODEL.h5")
optimizer = tf.keras.optimizers.Adam(1e-4)
mse = tf.keras.losses.MeanSquaredError()

def select_epsilon_greedy_action(state):
    return tf.argmax(main_nn(state)[0]).numpy()
num_episodes = 100



# Start training. Play game once and then train with a batch.
last_100_ep_rewards = []
loss_value = []
loss_value_1 = []
rew = []
ep_reward = 0
for episode in range(num_episodes+1):
  state = env.reset()
  ep_reward, done = 0, False
  loss = 0
  while not done:
    state_in = tf.expand_dims(state, axis=0)
    action = select_epsilon_greedy_action(state_in)
    next_state, reward, done = env.step(action)
    ep_reward += reward
    # Save to experience replay.
    state = next_state
    env.render_grid()
  
  if episode % 1 == 0:
    print(f'Episode {episode}/{num_episodes} '
          f'Reward in last 100 episodes: {ep_reward}')
    rew.append(reward)
    
plt.plot(rew)
plt.show()

        




