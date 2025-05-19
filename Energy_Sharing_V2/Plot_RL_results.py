# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 17:53:38 2021

@author: Suleman_Sahib
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
fsize = 20
"""
loss_DQN = np.load("LOSS_VALUE_TF_DQN_2.npy", allow_pickle=True)
loss_DDQN = np.load("LOSS_VALUE_TF_DDQN_2.npy", allow_pickle=True)
loss_Dueling_DQN = np.load("LOSS_VALUE_Duling_DQN_2.npy", allow_pickle=True)
loss_Dueling_DDQN = np.load("LOSS_VALUE_Duling_and_DDQN_2.npy", allow_pickle=True)
#loss_AC = np.load("Loss_AC.npy", allow_pickle=True)

plt.figure(figsize=((10,8)), dpi=300)
#fig, ax1 = plt.subplots(figsize=((10,10)), dpi=100)
#ax2 = ax1.twinx()
plt.plot(loss_DQN,"*-")
plt.plot(loss_DDQN,"p-")
plt.plot(loss_Dueling_DDQN,"^-")
plt.plot(loss_Dueling_DQN,"o-")
plt.xlabel("Episodes", fontsize=fsize)
plt.ylabel("Training Loss", fontsize=fsize)
plt.legend(["DQN", "DDQN", "Dueling DDQN", "Dueling DQN"], loc="upper right", fontsize=15)
plt.xlim(0,1400)
plt.xticks(np.arange(0,1401,200), fontsize=15)
plt.yticks(np.arange(0,700001,100000), fontsize=15)
plt.ylim(0,700000)
#ax1.tick_params(axis='y', labelsize=15)
#ax1.tick_params(axis='x', labelsize=15)
#ax2.tick_params(axis='y', labelsize=15)
plt.savefig("Loss_RL_300.png", transparent=True)
#plt.show()

"""

#reward_DQN = np.load("Reward_TF_DQN_2.npy", allow_pickle=True)
#reward_DDQN = np.load("Reward_TF_DDQN_2.npy", allow_pickle=True)
#reward_Dueling_DQN = np.load("Reward_Duling_DQN_2.npy", allow_pickle=True)
#reward_Dueling_DDQN = np.load("Reward_Duling_and_DDQN_2.npy", allow_pickle=True)

Epsilon_DQN = np.load("Epsilon_TF_DQN_2.npy", allow_pickle=True)


reward_DQN = np.load("DQN_Straight_Reward.npy", allow_pickle=True)
reward_DDQN = np.load("DDQN_Straight_Reward.npy", allow_pickle=True)
reward_Dueling_DQN = np.load("Duel_DQN_Straight_Reward.npy", allow_pickle=True)
reward_Dueling_DDQN = np.load("Duel_DDQN_Straight_Reward.npy", allow_pickle=True)


fig, ax1 = plt.subplots(figsize=((10,8)), dpi=300)
#ax2 = ax1.twinx() 

ax1.plot(reward_DQN, "^-")
ax1.plot(reward_DDQN, "--")
ax1.plot(reward_Dueling_DDQN, "-")
ax1.plot(reward_Dueling_DQN, "*-")
#Dueling_DQN_avg = [np.mean(reward_Dueling_DQN)] * len(reward_Dueling_DQN)
#ax1.plot(Dueling_DQN_avg, color='blue', linewidth=4)
#ax1.plot(reward_Dueling_DQN, "*-")
#x = np.arange(0,1400,1)
#ax1.plot(reward_Dueling_DQN.mean(axis=0),color="blue")

x = np.arange(0,1400,1)
y = reward_DQN
#linear_model=np.polyfit(x,reward_Dueling_DQN,2)
#linear_model_fn=np.poly1d(linear_model)
#ax1.plot(linear_model, color='blue', linewidth=4)

#z = np.polyfit(x, y, 3)
#p = np.poly1d(z)

with warnings.catch_warnings():
    warnings.simplefilter('ignore', np.RankWarning)
    p30 = np.poly1d(np.polyfit(x, y, 30))
xp = np.linspace(-1000, 1400, 100)
_ = ax1.plot( xp, p30(xp),  '*-', color='orange', linewidth=1.5, alpha=1.0)


x = np.arange(0,1400,1)
y = reward_DDQN
#linear_model=np.polyfit(x,reward_Dueling_DQN,2)
#linear_model_fn=np.poly1d(linear_model)
#ax1.plot(linear_model, color='blue', linewidth=4)

#z = np.polyfit(x, y, 3)
#p = np.poly1d(z)

with warnings.catch_warnings():
    warnings.simplefilter('ignore', np.RankWarning)
    p30 = np.poly1d(np.polyfit(x, y, 30))
xp = np.linspace(-1000, 1400, 100)
_ = ax1.plot( xp, p30(xp),  '*-', color='orange', linewidth=1.5, alpha=1.0)



x = np.arange(0,1400,1)
y = reward_Dueling_DDQN
#linear_model=np.polyfit(x,reward_Dueling_DQN,2)
#linear_model_fn=np.poly1d(linear_model)
#ax1.plot(linear_model, color='blue', linewidth=4)

#z = np.polyfit(x, y, 3)
#p = np.poly1d(z)

with warnings.catch_warnings():
    warnings.simplefilter('ignore', np.RankWarning)
    p30 = np.poly1d(np.polyfit(x, y, 30))
xp = np.linspace(-1000, 1400, 100)
_ = ax1.plot( xp, p30(xp),  '*-', color='orange', linewidth=1.5, alpha=1.0)



x = np.arange(0,1400,1)
y = reward_Dueling_DQN
#linear_model=np.polyfit(x,reward_Dueling_DQN,2)
#linear_model_fn=np.poly1d(linear_model)
#ax1.plot(linear_model, color='blue', linewidth=4)

#z = np.polyfit(x, y, 3)
#p = np.poly1d(z)

with warnings.catch_warnings():
    warnings.simplefilter('ignore', np.RankWarning)
    p30 = np.poly1d(np.polyfit(x, y, 30))
xp = np.linspace(-1000, 1400, 100)
_ = ax1.plot( xp, p30(xp), '*-', color='orange', linewidth=1.5, alpha=1.0)



ax1.set_ylabel("Reward", fontsize=fsize)
ax1.set_xlim(0,1400)
ax1.set_ylim(-1000,1100)

ax1.set_xlabel("Episodes", fontsize=fsize)

#ax1.plot(Epsilon_DQN, "b--")
#ax1.set_ylim(0,1.05)
#ax1.legend(["Epsilon"], bbox_to_anchor=(0.0, 1.0, 1.0, 0.13), loc="center", ncol=5, fontsize=15)

#, bbox_to_anchor=(0.0, 1.0, 1.0, 0.2), loc="center", ncol=1)
#ax1.legend(["DQN", "DDQN", "Dueling DQN", "Dueling DQN + DDQN", "Actor Critic"])
ax1.legend(["DQN", "DDQN", "Dueling DDQN", "Dueling DQN"], bbox_to_anchor=(0.0, 1.0, 1.0, 0.00), loc="center", ncol=5, fontsize=15)
#ax1.set_ylabel("Epsilon", fontsize=fsize)

ax1.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='x', labelsize=15)
#ax2.tick_params(axis='y', labelsize=12)
plt.savefig("D:\Energy_Sharing_V2\Reward_RL_300_straight.png", transparent=True)
#plt.show()



"""
Epsilon_DQN = np.load("Epsilon_TF_DQN.npy", allow_pickle=True)
Epsilon_DDQN = np.load("Epsilon_TF_DDQN.npy", allow_pickle=True)
Epsilon_Dueling_DQN = np.load("Epsilon_Duling_DQN.npy", allow_pickle=True)
Epsilon_Duling_and_DDQN = np.load("Epsilon_Duling_and_DDQN.npy", allow_pickle=True)
#Epsilon_AC = np.load("Epsilon_AC.npy", allow_pickle=True)


plt.figure(figsize=((10,8)))
plt.plot(Epsilon_DQN, "+")
plt.plot(Epsilon_DDQN, "--")
plt.plot(Epsilon_Dueling_DQN, "*")
plt.plot(Epsilon_Duling_and_DDQN, "-")
#plt.plot(Epsilon_AC,"^")
plt.plot(reward_Dueling_DDQN, "-")
plt.legend(["DQN", "DDQN", "Dueling DDQN", "Dueling DQN"])
plt.show()
"""