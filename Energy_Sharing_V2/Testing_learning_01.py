# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 15:20:11 2021

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

class dueling_DQN(tf.keras.Model):
  """Dense neural network class."""
  def __init__(self):
    super(dueling_DQN, self).__init__()
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

tf_dueling_dqn = dueling_DQN()
tf_dueling_ddqn = dueling_DQN()
#target_nn = dueling_DQN()


tf_dueling_dqn.build(input_shape=(None,num_features))
tf_dueling_ddqn.build(input_shape=(None,num_features))

tf_dueling_ddqn.load_weights("Duling_and_DDQN_TF_MODEL.h5")
tf_dueling_dqn.load_weights("Duling_DQN_TF_MODEL.h5")



class DQN(tf.keras.Model):
  """Dense neural network class."""
  def __init__(self):
    super(DQN, self).__init__()
    self.dense1 = tf.keras.layers.Dense(32, activation="relu")
    self.dense2 = tf.keras.layers.Dense(32, activation="relu")
    self.dense3 = tf.keras.layers.Dense(32, activation="relu")
    self.dense4 = tf.keras.layers.Dense(32, activation="relu")
    self.dense5 = tf.keras.layers.Dense(num_actions, dtype=tf.float32) # No activation
    
  def call(self, x):
    """Forward pass."""
    x = self.dense1(x)
    x = self.dense2(x)
    x = self.dense3(x)
    x = self.dense4(x)
    
    return self.dense5(x)

tf_ddqn = DQN()
tf_dqn = DQN()

tf_ddqn.build(input_shape=(None,num_features))
tf_dqn.build(input_shape=(None,num_features))

tf_dqn.load_weights("TF_DQN_MODEL.h5")
tf_ddqn.load_weights("TF_DDQN_MODEL.h5")


def DQN_action(state, eps):
    if eps > np.random.uniform(0,1):
        return env.action_space.sample()
    else:
        return tf.argmax(tf_dqn(state)[0]).numpy()
def DDQN_action(state, eps):
    if eps > np.random.uniform(0,1):
        return env.action_space.sample()
    else:
        return tf.argmax(tf_ddqn(state)[0]).numpy()
def Dueling_DDQN_action(state, eps):
    if eps > np.random.uniform(0,1):
        return env.action_space.sample()
    else:
        return tf.argmax(tf_dueling_ddqn(state)[0]).numpy()
def Dueling_DQN_action(state, eps):
    if eps > np.random.uniform(0,1):
        return env.action_space.sample()
    else:
        return tf.argmax(tf_dueling_dqn(state)[0]).numpy()


number_of_hours = 96
time_step = 96


num_episodes = 10000
epsilon = 1.0
start = 1.0
end = 0.0001
decay1 = 0.01
#
# 1000 episodes
epsilon_decay = 0.19 

bat_at_ship = []
bat_at_source= []
bat_at_source_load= []
bat_at_load = []
ep_reward = []
toy = []
actions_a = []

bat_shiped_SI = []
bat_shiped_SL = []
bat_received = []
price_invoice_SI = []
profit_gained_SI = []
price_invoice_SL = []
profit_gained_SL = []
deg = []
mt = []
re_util = []
actual_gen = []
#np.random.seed(99)
current_state = 0
last_state = 0
all_rewards = []
for hours in range(0,140000):
    done = False
    if hours < 2000:
        epsilon = 1.0
    else: 
        epsilon = 0.0000
    while not done:
        state_in = tf.expand_dims(state, axis=0)
       
        
        
        
        
        #if (hours == 20): # and (hours <= 28):
        #    env.source_island.export_battries()
        action = DQN_action(state_in, epsilon)
        #if (hours%5 == 0):
        #    action = env.action_space.sample()
        #else:
        #    action = Dueling_DQN_action(state_in) 
        
         #
        next_state, reward, done = env.step(action)
        #print(hours, reward)
        #actions_a.append(action)
        #env.render_grid()
      
    
    
        state = next_state
        if done:
            ep_reward.append(reward)
            
    #epsilon = end + (start - end) * math.exp(-1. * hours * decay1)
    
    
    #bat_shiped_SI.append(env.source_island.bat_shiped)
    #bat_shiped_SL.append(env.source_load.bat_shiped)
    
    #bat_received.append(env.load_island.bat_received)
    
    #price_invoice_SI.append(env.source_island.export_invoice)
    #price_invoice_SL.append(env.source_load.export_invoice)
    #profit_gained_SI.append(env.load_island.export_profit )
    #profit_gained_SL.append(env.load_island.export_profit)
    #current_state = 0
    #for bat in env.source_island.battries:
    #    current_state += bat.soc * 100/0.99
        
    
    #all_reward.append(reward)
    """
    actual_gen.append(env.source_island.current_re_gen)
    mt.append(env.source_island.mt_.generation_)
    deg.append(env.source_island.deg_.generation_)
    re_util.append(env.source_island.energy_utilized)
    #last_state += current_state 
    
    bat_at_ship.append(env.bat_at_ship*100)
    bat_at_source.append(env.source_island.battries_charged[-1]*100)
    bat_at_source_load.append(env.source_load.battries_charged[-1]*100)
    bat_at_load.append(env.load_island.battries_charged[-1]*100)
    """
    
    
    if len(ep_reward) >= 100:
        avg_rew = np.mean(ep_reward)
        ep_reward = []
        all_rewards.append(avg_rew)
        print(epsilon, avg_rew, hours)
    
    
plt.plot(all_rewards)
plt.show()
np.save("DQN_Straight_Reward.npy", all_rewards,allow_pickle=True)

"""
new_ = np.asarray(actual_gen) - np.asarray(re_util)
fontsize = 18
fonttick = 18
plt.figure(figsize=((15,8)), dpi=300)
plt.subplot(2,1,1)
#plt.plot(new_, "r*-")
plt.plot(actual_gen, "*-")
plt.plot(re_util, "^-")
plt.plot(mt, "o-")
plt.plot(deg, "--", linewidth=5)
plt.legend(["RE Gen", "RE Utilized", "MT", "DEG"], fontsize=fontsize, loc="upper left")
plt.xticks([])
plt.ylabel("kW",fontsize=fontsize )
plt.yticks(fontsize=fonttick)
plt.xlim(0,96)

#plt.yticks(np.arange(0,1001,200), fontsize=fonttick)

ax1 = plt.subplot(2,1,2)
ax2 = ax1.twinx()
ax1.plot(new_, "g*-")
ax1.legend(["RE Curtail"], loc="upper left", bbox_to_anchor=(0.0, 1.0, 1.0, 0.0), fontsize=fontsize)
ax2.plot(bat_at_source, "o-")
ax2.legend(["SI Storage"], loc="upper left", bbox_to_anchor=(0.0, 0.85,1.0, 0.0), fontsize=fontsize)

ax1.set_yticklabels(np.arange(0,201, 50), fontsize=fonttick)
ax2.set_yticklabels(np.arange(0,1001, 200), fontsize=fonttick)
ax2.set_yticks(np.arange(0,1001, 200))
ax1.set_yticks(np.arange(0,201, 50))
ax1.set_xticklabels(np.arange(0,time_step+1,24), fontsize=fonttick)

ax1.set_xticks(np.arange(0,time_step+1,24))
ax1.set_xlabel("Time Hours", fontsize=fontsize)
ax1.set_ylabel('kW', fontsize=fontsize)
ax2.set_ylabel('kWh', fontsize=fontsize)

ax1.set_xlim(0,96)
ax2.set_xlim(0,96)
plt.savefig("E:/Energy_Sharing_V2/Results_Plots/RE_Utilization.png", transparent=True)
#plt.show()


fontsize = 15
fonttick = 12
plt.figure(figsize=((15,8)), dpi=300)
plt.subplot(3,1,1)
plt.plot(bat_at_source, "b*-")
plt.legend(["SI"], fontsize=fontsize, loc="upper left")
plt.xticks([])
plt.yticks(fontsize=fonttick)
plt.xlim(0,96)

plt.yticks(np.arange(0,1001,200), fontsize=fonttick)

plt.subplot(3,1,2)
plt.plot(bat_at_ship, "g*-")
plt.xticks([])
plt.legend(["SHIP"], fontsize=fontsize, loc="upper left")
plt.ylabel("Storage Cluster Cap (kWh)", fontsize=fontsize)
plt.yticks(np.arange(0,1001,200), fontsize=fonttick)
plt.xlim(0,96)

plt.subplot(3,1,3)
plt.plot(bat_at_load, "r*-")
plt.xticks(np.arange(0,time_step+1,24), fontsize=fonttick)
plt.legend(["LIN"], fontsize=fontsize, loc="upper left")

plt.yticks(np.arange(0,1001,200), fontsize=fonttick)
plt.xlim(0,96)
plt.xlabel("Time Hours", fontsize=fontsize)

plt.savefig("E:/Energy_Sharing_V2/Results_Plots/Storage_capacity.png", transparent=True)
#plt.show()
"""





"""
ax1 = plt.figure(figsize=((15,8)), dpi=100)
ax1 = plt.subplot(1,1,1)
#plt.subplots(figsize=((10,3)), dpi=100)
ax2 = ax1.twinx()
ax1.bar(env.load_island.all_time_steps[0:96], bat_at_load, color = "green")
ax1.plot(bat_at_ship, "b*-")
ax1.plot(bat_at_source, "r*-")
#ax1.plot(bat_at_source_load, "*-")
ax2.bar(env.load_island.mt_.time_step, env.load_island.mt_.gen_plt,  width=0.5, bottom=0, align='center')
ax2.bar(env.load_island.deg_.time_step,env.load_island.deg_.gen_plt,  width=0.5, bottom=10, align='edge')


ax1.set_ylim(0,9)
#ax2.set_xlim(0,time_step)
#ax2.set_ylim(0,41)
ax2.set_yticks(np.arange(0,51, 25))
ax1.set_yticks(np.arange(0,11, 5))

#ax1.set_xticks(np.arange(0,time_step+1,24))

ax1.set_xlabel("Time Hours", fontsize=15)
plt.show()
"""


"""
np.save("1action_state_attack_20_bat_at_source.npy", bat_at_source, allow_pickle=True)
np.save("1action_state_attack_20_all_time_steps.npy", env.load_island.all_time_steps, allow_pickle=True)
np.save("1action_state_attack_20_bat_at_load.npy",bat_at_load  , allow_pickle=True)
np.save("1action_state_attack_20_mt_time_step.npy",env.load_island.mt_.time_step  , allow_pickle=True)
np.save("1action_state_attack_20_mt_gen.npy",env.load_island.mt_.gen_plt  , allow_pickle=True)
np.save("1action_state_attack_20_deg_time_step.npy", env.load_island.deg_.time_step , allow_pickle=True)
np.save("1action_state_attack_20_deg_gen.npy", env.load_island.deg_.gen_plt , allow_pickle=True)
np.save("1action_state_attack_20_bat_at_ship.npy", bat_at_ship, allow_pickle=True)
"""


"""
np.save("action_attack_hf5_bat_at_source.npy", bat_at_source, allow_pickle=True)
np.save("action_attack_hf5_all_time_steps.npy", env.load_island.all_time_steps, allow_pickle=True)
np.save("action_attack_hf5_bat_at_load.npy",bat_at_load  , allow_pickle=True)
np.save("action_attack_hf5_mt_time_step.npy",env.load_island.mt_.time_step  , allow_pickle=True)
np.save("action_attack_hf5_mt_gen.npy",env.load_island.mt_.gen_plt  , allow_pickle=True)
np.save("action_attack_hf5_deg_time_step.npy", env.load_island.deg_.time_step , allow_pickle=True)
np.save("action_attack_hf5_deg_gen.npy", env.load_island.deg_.gen_plt , allow_pickle=True)
np.save("action_attack_hf5_bat_at_ship.npy", bat_at_ship, allow_pickle=True)
"""


"""   
np.save("Dueling_DQN_Island_Bat_utilization.npy", env.load_island.battries_charged, allow_pickle=True )

np.save("Dueling_DQN_Island_Bat_at_ship.npy", bat_at_ship, allow_pickle=True)

np.save("Dueling_DQN_Island_MT.npy", env.load_island.mt_.gen_plt, allow_pickle=True)

np.save("Dueling_DQN_Island_DEG.npy", env.load_island.deg_.gen_plt, allow_pickle=True) 
   
np.save("Dueling_DQN_Island_MT_Time.npy", env.load_island.mt_.time_step, allow_pickle=True)
np.save("Dueling_DQN_Island_DEG_Time.npy", env.load_island.deg_.time_step, allow_pickle=True)
np.save("Dueling_DQN_All_Time.npy",env.load_island.all_time_steps, allow_pickle=True)
"""
#print(len(env.source_island.total_cost))
#print(len(env.source_load.total_cost))
#np.save("DQN_export_profit .npy",env.export_profit , allow_pickle=True)
#np.save("DQN_cost_of_islands.npy",env.cost_of_islands, allow_pickle=True)
#np.save("DQN_battries_delivered .npy",env.battries_delivered ,allow_pickle=True)

#np.save("case_100_bat_at_source.npy",bat_at_source, allow_pickle=True)
#np.save("case_100_bat_at_source_load.npy",bat_at_source_load, allow_pickle=True)
#np.save("case_100_bat_at_laod.npy",bat_at_load,allow_pickle=True)
#np.save("case_100_bat_at_ship.npy",bat_at_ship,allow_pickle=True)

#start = 0
#time_step = 160 

"""
plt.figure(figsize=((30,4))) #, dpi=100)
#plt.bar(env.source_island.all_time_steps[0:time_step], env.source_island.battries_charged[0:time_step])#, color = "green")
plt.plot(env.source_island.battries_charged[0:time_step])#, color = "green")
#, color = "green")
plt.bar(np.arange(1,len(env.export_profit)+1,1), env.export_profit,  width=0.5, align='edge')#, color = "green")
plt.bar(np.arange(1,len(env.cost_of_islands)+1,1), env.cost_of_islands,  width=0.3, align='center')#, color = "green")

#plt.plot(bat_at_ship[start:time_step], "r*-")
#plt.bar(env.source_island.all_time_steps[0:time_step], bat_at_ship[0:time_step])
#plt.draw( )
#plt.xticks(np.arange(0,time_step,1), labels=np.arange(20,46,1))
plt.xticks(np.arange(1,len(env.export_profit)+1,1))
#plt.xlim(0,time_step)
#plt.ylim(0,10)
plt.legend(["BS","CS"] , bbox_to_anchor=(0.0, 1.0, 1.0, 0.1),loc="center", ncol=4, fontsize=12)
plt.grid()

plt.ylabel("Profit $", fontsize=15)
plt.xlabel("Dispatched Delivery (LIN)", fontsize=15)
#plt.savefig("Attck_600.png", transparent=True)
plt.show()
#plt.plot(env.source_island.battries[0].soc_plt[0:24])
#plt.bar(env.source_island.all_time_steps[0:24], actions_a[0:24])
#
"""
"""
plt.figure(figsize=((15,8)), dpi=100)
#plt.bar(env.source_island.all_time_steps[0:time_step], env.source_island.battries_charged[0:time_step])#, color = "green")
#plt.plot(env.source_island.battries_charged[start:time_step], "bp-")#, color = "green")
#, color = "green")
#plt.plot(env.source_load.battries_charged[start:time_step], "^-")
#plt.plot(env.load_island.battries_charged[start:time_step],"g-")#, color = "green")
#, color = "green")
plt.plot(bat_at_source[start:time_step], "bp-")
plt.plot(bat_at_source_load[start:time_step], "^-")
plt.plot(bat_at_load[start:time_step],"g-")#, color = "
plt.plot(bat_at_ship[start:time_step], "r*-")
#plt.bar(env.source_island.all_time_steps[0:time_step], bat_at_ship[0:time_step])
#plt.draw( )
#plt.xticks(np.arange(start,time_step,1), labels=np.arange(20,46,1))
#plt.yticks(np.arange(0,11,1))
#plt.xlim(20,48)
#plt.ylim(0,10)
plt.legend(["SI","SL","LIN","SHIP"] , bbox_to_anchor=(0.0, 1.0, 1.0, 0.1),loc="center", ncol=4, fontsize=12)
plt.grid()

plt.ylabel("No. of Batteries", fontsize=15)
plt.xlabel("Time Hours", fontsize=15)
#plt.savefig("Attck_600.png", transparent=True)
plt.show()

"""


"""
"""

"""
#plt.figure(figsize=((10,10)), dpi=100)
#ax1 = plt.subplot(2,2,2)
fig, ax1 = plt.subplots(figsize=((10,3)), dpi=100)
ax2 = ax1.twinx()
ax1.bar(env_1.load_island.all_time_steps[0:time_step], env_1.load_island.battries_charged[0:time_step], color = "green")
ax1.plot(bat_at_ship_1, "r--")
ax2.bar(env_1.load_island.mt_.time_step, env_1.load_island.mt_.gen_plt,  width=0.5, bottom=0, align='center',)
ax2.bar(env_1.load_island.deg_.time_step, env_1.load_island.deg_.gen_plt,  width=0.5, bottom=10, align='edge',)

ax1.set_xlim(0,time_step)
ax1.set_ylim(0,9)
ax2.set_xlim(0,time_step)


#ax1 = plt.subplot(2,2,3)
fig, ax1 = plt.subplots(figsize=((10,3)), dpi=100)
ax2 = ax1.twinx()
ax1.bar(env_2.load_island.all_time_steps[0:time_step], env_2.load_island.battries_charged[0:time_step], color = "green")
ax1.plot(bat_at_ship_2, "r--")
ax2.bar(env_2.load_island.mt_.time_step, env_2.load_island.mt_.gen_plt,  width=0.5, bottom=0, align='center',)
ax2.bar(env_2.load_island.deg_.time_step, env_2.load_island.deg_.gen_plt,  width=0.5, bottom=10, align='edge',)

ax1.set_xlim(0,time_step)
ax1.set_ylim(0,9)
ax2.set_xlim(0,time_step)

#ax1 = plt.subplot(2,2,4)
fig, ax1 = plt.subplots(figsize=((10,3)), dpi=100)
ax2 = ax1.twinx()
ax1.bar(env_3.load_island.all_time_steps[0:time_step], env_3.load_island.battries_charged[0:time_step], color = "green")
ax1.plot(bat_at_ship_3, "r--")
ax2.bar(env_3.load_island.mt_.time_step, env_3.load_island.mt_.gen_plt,  width=0.5, bottom=0, align='center',)
ax2.bar(env_3.load_island.deg_.time_step, env_3.load_island.deg_.gen_plt,  width=0.5, bottom=10, align='edge',)

ax1.set_xlim(0,time_step)
ax1.set_ylim(0,9)
ax2.set_xlim(0,time_step)

plt.xticks(np.arange(0,time_step+1,24))
"""


"""

number_of_hours = 170



# Start training. Play game once and then train with a batch.
last_100_ep_rewards = []
loss_value = []
loss_value_1 = []
rew = []
bat_at_ship = []
ep_reward = 0
toy = []
    
for hours in range(1,number_of_hours):
    state_in = tf.expand_dims(state, axis=0)
    action = DDQN_action(state_in)
    next_state, reward, done = env_1.step(action)
    ep_reward += reward
    # Save to experience replay.
    state = next_state
    bat_at_ship.append(env.bat_at_ship)

    rew.append(reward)

ax1 = plt.subplot(2,2,2)
#fig, ax1 = plt.subplots(figsize=((10,10)), dpi=100)
ax2 = ax1.twinx()
ax1.bar(env.load_island.all_time_steps[0:time_step], env.load_island.battries_charged[0:time_step], color = "green")
ax1.plot(bat_at_ship, "r--")
ax2.bar(env.load_island.mt_.time_step, env.load_island.mt_.gen_plt,  width=0.5, bottom=0, align='center',)
ax2.bar(env.load_island.deg_.time_step, env.load_island.deg_.gen_plt,  width=0.5, bottom=10, align='edge',)

ax1.set_xlim(0,time_step)
ax1.set_ylim(0,9)
ax2.set_xlim(0,time_step)
#ax2.set_ylim(0,35)
plt.xticks(np.arange(0,time_step+1,24))



number_of_hours = 170



# Start training. Play game once and then train with a batch.
last_100_ep_rewards = []
loss_value = []
loss_value_1 = []
rew = []
bat_at_ship = []
ep_reward = 0
toy = []

    
for hours in range(1,number_of_hours):
    state_in = tf.expand_dims(state, axis=0)
    action = Dueling_DDQN_action(state_in)
    next_state, reward, done = env_2.step(action)
    ep_reward += reward
    # Save to experience replay.
    state = next_state
    bat_at_ship.append(env.bat_at_ship)

    rew.append(reward)

ax1 = plt.subplot(2,2,3)
#fig, ax1 = plt.subplots(figsize=((10,10)), dpi=100)
ax2 = ax1.twinx()
ax1.bar(env.load_island.all_time_steps[0:time_step], env.load_island.battries_charged[0:time_step], color = "green")
ax1.plot(bat_at_ship, "r--")
ax2.bar(env.load_island.mt_.time_step, env.load_island.mt_.gen_plt,  width=0.5, bottom=0, align='center',)
ax2.bar(env.load_island.deg_.time_step, env.load_island.deg_.gen_plt,  width=0.5, bottom=10, align='edge',)

ax1.set_xlim(0,time_step)
ax1.set_ylim(0,9)
ax2.set_xlim(0,time_step)
#ax2.set_ylim(0,35)
plt.xticks(np.arange(0,time_step+1,24))






number_of_hours = 170



# Start training. Play game once and then train with a batch.
last_100_ep_rewards = []
loss_value = []
loss_value_1 = []
rew = []
bat_at_ship = []
ep_reward = 0
toy = []
   
for hours in range(1,number_of_hours):
    state_in = tf.expand_dims(state, axis=0)
    action = Dueling_DQN_action(state_in)
    next_state, reward, done = env_3.step(action)
    ep_reward += reward
    # Save to experience replay.
    state = next_state
    bat_at_ship.append(env.bat_at_ship)

    rew.append(reward)
"""



"""
# Battries Charged at Island    
plt.figure(figsize=(10,5))
plt.plot(env.source_island.battries_charged[0:time_step])
plt.xlim(0,time_step)
plt.xticks(np.arange(0,time_step+1,1))
plt.show()

plt.figure(figsize=(10,5))
plt.plot(env.source_load.battries_charged[0:time_step])
plt.xlim(0,time_step)
plt.xticks(np.arange(0,time_step+1,1))
plt.show()
"""
"""
plt.figure(figsize=(10,5))
plt.plot(env.load_island.battries_charged[0:time_step])
plt.xlim(0,time_step)
plt.xticks(np.arange(0,time_step+1,1))
plt.show()
"""
"""
plt.figure(figsize=(10,5))
plt.plot(bat_at_ship)
plt.xlim(0,time_step)
plt.xticks(np.arange(0,time_step+1,1))
plt.show()
#print(len(env.load_island.deg_.gen_plt), len(env.load_island.mt_.gen_plt))

plt.figure(figsize=((10,10)), dpi=100)
plt.subplot(2,2,1)
plt.plot(env.load_island.battries_charged[0:time_step],'r')
plt.xlim(0,time_step)
plt.ylim(0,9)

plt.subplot(2,2,2)
plt.plot(bat_at_ship, 'b--')

plt.xlim(0,time_step)
plt.ylim(0,9)

plt.subplot(2,2,3)
plt.bar(env.load_island.mt_.time_step, env.load_island.mt_.gen_plt, color = 'yellow')

plt.xlim(0,time_step)
plt.ylim(0,10)
 
plt.subplot(2,2,4)
plt.bar(env.load_island.deg_.time_step, env.load_island.deg_.gen_plt, color = 'green')
plt.xlim(0,time_step)
plt.show()
"""   
    
"""
ax1 = plt.subplot(2,2,4)
#fig, ax1 = plt.subplots(figsize=((10,10)), dpi=100)
ax2 = ax1.twinx()
ax1.bar(env.load_island.all_time_steps[0:time_step], env.load_island.battries_charged[0:time_step], color = "green")
ax1.plot(bat_at_ship, "r--")
ax2.bar(env.load_island.mt_.time_step, env.load_island.mt_.gen_plt,  width=0.5, bottom=0, align='center',)
ax2.bar(env.load_island.deg_.time_step, env.load_island.deg_.gen_plt,  width=0.5, bottom=10, align='edge',)

ax1.set_xlim(0,time_step)
ax1.set_ylim(0,9)
ax2.set_xlim(0,time_step)
#ax2.set_ylim(0,35)
plt.xticks(np.arange(0,time_step+1,24))
plt.show()

"""


        




