# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 20:53:23 2021

@author: Suleman_Sahib
"""

import numpy as np 
import matplotlib.pyplot as plt
time_step = 96
dpi = 300
fontsize = 18
fontsize1 = 20

"""
bat_at_source = np.load("case_100_bat_at_source.npy", allow_pickle=True)
bat_at_source_load = np.load("case_100_bat_at_source_load.npy", allow_pickle=True)
bat_at_load = np.load("case_100_bat_at_laod.npy",allow_pickle=True)
bat_at_ship = np.load("case_100_bat_at_ship.npy",allow_pickle=True)

plt.figure(figsize=((15,8)), dpi=dpi)
plt.subplot(3,1,1)

plt.plot(bat_at_source, "*-")
plt.plot(bat_at_source_load, "^-")
plt.plot(bat_at_load, "o-")
plt.plot(bat_at_ship, "p-")


plt.ylim(0,10)
plt.xlim(0,time_step)
plt.yticks(np.arange(0,11, 2), fontsize=fontsize1)
plt.xticks([])
#plt.xlabel("Time Hours", fontsize=fontsize)
#plt.ylabel("No. of Batteries", fontsize=fontsize)

plt.legend(["Bat_SI","Bat_SL","Bat_LIN","Bat_SHIP"] , bbox_to_anchor=(0.0, 1.0, 1.0, 0.2),loc="center", ncol=4, fontsize=fontsize)
#plt.grid()
plt.axvspan(20, 96, color='yellow', alpha=0.3)

bat_at_source = np.load("case_80_bat_at_source.npy", allow_pickle=True)
bat_at_source_load = np.load("case_80_bat_at_source_load.npy", allow_pickle=True)
bat_at_load = np.load("case_80_bat_at_laod.npy",allow_pickle=True)
bat_at_ship = np.load("case_80_bat_at_ship.npy",allow_pickle=True)

plt.subplot(3,1,2)

plt.plot(bat_at_source, "*-")
plt.plot(bat_at_source_load, "^-")
plt.plot(bat_at_load, "o-")
plt.plot(bat_at_ship, "p-")


plt.ylim(0,10)
plt.xlim(0,time_step)
plt.yticks(np.arange(0,11, 2), fontsize=fontsize1)
plt.xticks([])
plt.axvspan(44, 96, color='yellow', alpha=0.3)
plt.ylabel("No. of Batteries", fontsize=fontsize)

bat_at_source = np.load("case_50_bat_at_source.npy", allow_pickle=True)
bat_at_source_load = np.load("case_50_bat_at_source_load.npy", allow_pickle=True)
bat_at_load = np.load("case_50_bat_at_laod.npy",allow_pickle=True)
bat_at_ship = np.load("case_50_bat_at_ship.npy",allow_pickle=True)

plt.subplot(3,1,3)

plt.plot(bat_at_source, "*-")
plt.plot(bat_at_source_load, "^-")
plt.plot(bat_at_load, "o-")
plt.plot(bat_at_ship, "p-")


plt.ylim(0,10)
plt.xlim(0,time_step)
plt.yticks(np.arange(0,11, 2), fontsize=fontsize1)
plt.xticks(np.arange(0,time_step+1,24), fontsize=fontsize1)
plt.xlabel("Time Hours", fontsize=fontsize)

plt.subplots_adjust(hspace=0.1)

#plt.legend(["SI","SL","LIN","SHIP"] , bbox_to_anchor=(0.0, 1.0, 1.0, 0.1),loc="center", ncol=4, fontsize=fontsize)
#plt.grid()

#plt.savefig("Attck_600.png", transparent=True)

plt.axvspan(68, 96, color='yellow', alpha=0.3)




#plt.subplots_adjust(bottom=0.0, top=0.7, hspace=0.1)



plt.text(0.4, 29, 'RE_Curt =\n 0%', fontsize=fontsize)
plt.text(0.4, 19, 'RE_Curt = 20%', fontsize=fontsize)
plt.text(0.4, 8, 'RE_Curt = 50%', fontsize=fontsize)


plt.savefig("RE_Curtail_Case_300.png", transparent=True)

#plt.show()


"""

"""

# State Action Attack
bat_at_source = np.load("1action_state_attack_20_bat_at_source.npy", allow_pickle=True)
all_time_steps=np.load("1action_state_attack_20_all_time_steps.npy",  allow_pickle=True)
bat_at_load=np.load("1action_state_attack_20_bat_at_load.npy",allow_pickle=True)
mt_time_step=np.load("1action_state_attack_20_mt_time_step.npy", allow_pickle=True)
mt_gen=np.load("1action_state_attack_20_mt_gen.npy", allow_pickle=True)
deg_time_step=np.load("1action_state_attack_20_deg_time_step.npy", allow_pickle=True)
deg_gen=np.load("1action_state_attack_20_deg_gen.npy",  allow_pickle=True)

bat_at_ship=np.load("1action_state_attack_20_bat_at_ship.npy",  allow_pickle=True)






plt.figure(figsize=((10,6)), dpi=dpi)
ax1 = plt.subplot(2,1,1)
#ax1 =plt.subplots(figsize=((10,6)), dpi=100)
ax1.set_xticks([])
ax2 = ax1.twinx()
ax1.bar(all_time_steps[0:96], bat_at_load, color = "green")
ax1.plot(bat_at_ship, "b*-")
ax1.plot(bat_at_source, "r*-")
ax2.bar(mt_time_step, mt_gen,  width=0.5, bottom=0, align='center')
ax2.bar(deg_time_step,deg_gen,  width=0.5, bottom=10, align='edge')


ax1.set_ylim(0,10)
ax1.set_xlim(0,time_step)
ax2.set_ylim(0,41)
ax2.set_yticks(np.arange(0,51, 10))
ax1.set_yticks(np.arange(0,11, 2))

ax1.legend(["Battery_LIN","Battery_Ship","Battery_SI"], bbox_to_anchor=(0.0, 1.0, 0.65, 0.15),loc="center", ncol=5, fontsize=12)
ax2.legend(["MT_LIN","DEG_LIN"], bbox_to_anchor=(0.83, 1.0, 0.0, 0.15),loc="center", ncol=5, fontsize=12)


ax1.set_ylabel("No. of Batteries", fontsize=fontsize)
ax2.set_ylabel("Power KW", fontsize=fontsize)


ax1.set_xticks(np.arange(0,time_step+1,24))
ax1.set_xlabel("Time Hours", fontsize=fontsize)

#plt.subplots_adjust(bottom=0.0, top=0.7, hspace=0.1)
plt.axvspan(18, 22, color='yellow', alpha=0.3)
plt.axvspan(38, 42, color='yellow', alpha=0.3)
plt.axvspan(58, 62, color='yellow', alpha=0.3)
plt.axvspan(78, 82, color='yellow', alpha=0.3)


#plt.text(0.4, 100, 'State & Action = 20 hrs', fontsize=12)
plt.text(0.4, 46, 'State & Action = 20 hrs', fontsize=fontsize)

plt.savefig("State_Action_Atack_300.png", transparent=True)

#plt.show()



"""
"""
# State Attack 

bat_at_source = np.load("state_attack_one20_bat_at_source.npy", allow_pickle=True)
all_time_steps=np.load("state_attack_one20_all_time_steps.npy",  allow_pickle=True)
bat_at_load=np.load("state_attack_one20_bat_at_load.npy",allow_pickle=True)
mt_time_step=np.load("state_attack_one20_mt_time_step.npy", allow_pickle=True)
mt_gen=np.load("state_attack_one20_mt_gen.npy", allow_pickle=True)
deg_time_step=np.load("state_attack_one20_deg_time_step.npy", allow_pickle=True)
deg_gen=np.load("state_attack_one20_deg_gen.npy",  allow_pickle=True)

bat_at_ship=np.load("state_attack_one20_bat_at_ship.npy",  allow_pickle=True)






plt.figure(figsize=((15,8)), dpi=dpi)
ax1 = plt.subplot(2,1,1)
#plt.subplots(figsize=((10,3)), dpi=100)
ax1.set_xticks([])
ax2 = ax1.twinx()
ax1.bar(all_time_steps[0:96], bat_at_load, color = "green")
ax1.plot(bat_at_ship, "b*-")
ax1.plot(bat_at_source, "r*-")
ax2.bar(mt_time_step, mt_gen,  width=0.5, bottom=0, align='center')
ax2.bar(deg_time_step,deg_gen,  width=0.5, bottom=10, align='edge')


ax1.set_ylim(0,10)
ax1.set_xlim(0,time_step)
ax2.set_ylim(0,41)
ax2.set_yticks(np.arange(0,51, 10))
ax1.set_yticks(np.arange(0,11, 2))
ax1.tick_params(axis='y', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)

ax1.legend(["Battery_LIN","Battery_Ship","Battery_SI"], bbox_to_anchor=(0.1, 1.0, 0.5, 0.15),loc="center", ncol=5, fontsize=15)
ax2.legend(["MT_LIN","DEG_LIN"], bbox_to_anchor=(0.85, 1.0, 0.0, 0.15),loc="center", ncol=5, fontsize=15)


ax1.set_ylabel("No. of Batteries", loc = "bottom", fontsize=fontsize)
ax2.set_ylabel("Power KW", loc = "bottom", fontsize=fontsize)
#ax1.set_xticks(np.arange(0,time_step+1,24))
plt.axvspan(18, 22, color='yellow', alpha=0.3)


bat_at_source = np.load("state_attack_multi_bat_at_source.npy", allow_pickle=True)
all_time_steps=np.load("state_attack_multi_all_time_steps.npy",  allow_pickle=True)
bat_at_load=np.load("state_attack_multi_bat_at_load.npy",allow_pickle=True)
mt_time_step=np.load("state_attack_multi_mt_time_step.npy", allow_pickle=True)
mt_gen=np.load("state_attack_multi_mt_gen.npy", allow_pickle=True)
deg_time_step=np.load("state_attack_multi_deg_time_step.npy", allow_pickle=True)
deg_gen=np.load("state_attack_multi_deg_gen.npy",  allow_pickle=True)

bat_at_ship=np.load("state_attack_multi_bat_at_ship.npy",  allow_pickle=True)


ax1 = plt.subplot(2,1,2)
#plt.subplots(figsize=((10,3)), dpi=dpi)
ax2 = ax1.twinx()
ax1.bar(all_time_steps[0:96], bat_at_load, color = "green")
ax1.plot(bat_at_ship, "b*-")
ax1.plot(bat_at_source, "r*-")
ax2.bar(mt_time_step, mt_gen,  width=0.5, bottom=0, align='center')
ax2.bar(deg_time_step,deg_gen,  width=0.5, bottom=10, align='edge')


ax1.set_ylim(0,10)
ax1.set_xlim(0,time_step)
ax2.set_ylim(0,41)
ax2.set_yticks(np.arange(0,51, 10))
ax1.set_yticks(np.arange(0,11, 2))

ax1.set_xticks(np.arange(0,time_step+1,24))

ax1.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)
plt.subplots_adjust(hspace=0.1)
plt.axvspan(18, 22, color='yellow', alpha=0.3)
plt.axvspan(38, 42, color='yellow', alpha=0.3)
plt.axvspan(58, 62, color='yellow', alpha=0.3)
plt.axvspan(78, 82, color='yellow', alpha=0.3)

ax1.set_xlabel("Time Hours", fontsize=fontsize)

plt.text(0.4, 100, 'State = 20 hrs', fontsize=fontsize)
plt.text(0.4, 45, 'State = 20 hrs', fontsize=fontsize)

plt.savefig("State_Atack_300.png", transparent=True)

#plt.show()
    



# Action Attack Ploting

bat_at_source = np.load("action_attack_lf10_bat_at_source.npy", allow_pickle=True)
all_time_steps=np.load("action_attack_lf10_all_time_steps.npy",  allow_pickle=True)
bat_at_load=np.load("action_attack_lf10_bat_at_load.npy",allow_pickle=True)
mt_time_step=np.load("action_attack_lf10_mt_time_step.npy", allow_pickle=True)
mt_gen=np.load("action_attack_lf10_mt_gen.npy", allow_pickle=True)
deg_time_step=np.load("action_attack_lf10_deg_time_step.npy", allow_pickle=True)
deg_gen=np.load("action_attack_lf10_deg_gen.npy",  allow_pickle=True)

bat_at_ship=np.load("action_attack_lf10_bat_at_ship.npy",  allow_pickle=True)



plt.figure(figsize=((15,8)), dpi=dpi)
ax1 = plt.subplot(2,1,1)
#plt.subplots(figsize=((10,3)), dpi=100)
ax1.set_xticks([])
ax2 = ax1.twinx()
ax1.bar(all_time_steps[0:96], bat_at_load, color = "green")
ax1.plot(bat_at_ship, "b*-")
ax1.plot(bat_at_source, "r*-")
ax2.bar(mt_time_step, mt_gen,  width=0.5, bottom=0, align='center')
ax2.bar(deg_time_step,deg_gen,  width=0.5, bottom=10, align='edge')


ax1.set_ylim(0,10)
ax1.set_xlim(0,time_step)
ax2.set_ylim(0,41)
ax2.set_yticks(np.arange(0,51, 10))
ax1.set_yticks(np.arange(0,11, 2))
ax1.tick_params(axis='y', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)
ax1.legend(["Battery_LIN","Battery_Ship","Battery_SI"], bbox_to_anchor=(0.1, 1.0, 0.5, 0.15),loc="center", ncol=5, fontsize=15)
ax2.legend(["MT_LIN","DEG_LIN"], bbox_to_anchor=(0.85, 1.0, 0.0, 0.15),loc="center", ncol=5, fontsize=15)


ax1.set_ylabel("No. of Batteries", loc = "bottom", fontsize=fontsize)
ax2.set_ylabel("Power KW", loc = "bottom", fontsize=fontsize)
#ax1.set_xticks(np.arange(0,time_step+1,24))
plt.axvspan(8, 12, color='yellow', alpha=0.3)
plt.axvspan(18, 22, color='yellow', alpha=0.3)
plt.axvspan(28, 32, color='yellow', alpha=0.3)
plt.axvspan(38, 42, color='yellow', alpha=0.3)
plt.axvspan(48, 52, color='yellow', alpha=0.3)
plt.axvspan(58, 62, color='yellow', alpha=0.3)
plt.axvspan(68, 72, color='yellow', alpha=0.3)
plt.axvspan(78, 82, color='yellow', alpha=0.3)
plt.axvspan(88, 92, color='yellow', alpha=0.3)



bat_at_source = np.load("action_attack_hf5_bat_at_source.npy", allow_pickle=True)
all_time_steps=np.load("action_attack_hf5_all_time_steps.npy",  allow_pickle=True)
bat_at_load=np.load("action_attack_hf5_bat_at_load.npy",allow_pickle=True)
mt_time_step=np.load("action_attack_hf5_mt_time_step.npy", allow_pickle=True)
mt_gen=np.load("action_attack_hf5_mt_gen.npy", allow_pickle=True)
deg_time_step=np.load("action_attack_hf5_deg_time_step.npy", allow_pickle=True)
deg_gen=np.load("action_attack_hf5_deg_gen.npy",  allow_pickle=True)

bat_at_ship=np.load("action_attack_hf5_bat_at_ship.npy",  allow_pickle=True)

ax1 = plt.subplot(2,1,2)
#plt.subplots(figsize=((10,3)), dpi=dpi)
ax2 = ax1.twinx()
ax1.bar(all_time_steps[0:96], bat_at_load, color = "green")
ax1.plot(bat_at_ship, "b*-")
ax1.plot(bat_at_source, "r*-")
ax2.bar(mt_time_step, mt_gen,  width=0.5, bottom=0, align='center')
ax2.bar(deg_time_step,deg_gen,  width=0.5, bottom=10, align='edge')


ax1.set_ylim(0,10)
ax1.set_xlim(0,time_step)
ax2.set_ylim(0,41)
ax2.set_yticks(np.arange(0,51, 10))
ax1.set_yticks(np.arange(0,11, 2))

ax1.set_xticks(np.arange(0,time_step+1,24))
ax1.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)
plt.subplots_adjust(hspace=0.1)
plt.axvspan(4, 6, color='yellow', alpha=0.3)
plt.axvspan(9, 11, color='yellow', alpha=0.3)
plt.axvspan(14, 16, color='yellow', alpha=0.3)
plt.axvspan(19, 21, color='yellow', alpha=0.3)
plt.axvspan(24, 26, color='yellow', alpha=0.3)
plt.axvspan(29, 31, color='yellow', alpha=0.3)
plt.axvspan(34, 36, color='yellow', alpha=0.3)
plt.axvspan(39, 41, color='yellow', alpha=0.3)
plt.axvspan(44, 46, color='yellow', alpha=0.3)
plt.axvspan(49, 51, color='yellow', alpha=0.3)
plt.axvspan(54, 56, color='yellow', alpha=0.3)
plt.axvspan(59, 61, color='yellow', alpha=0.3)
plt.axvspan(64, 66, color='yellow', alpha=0.3)
plt.axvspan(69, 71, color='yellow', alpha=0.3)
plt.axvspan(74, 76, color='yellow', alpha=0.3)
plt.axvspan(79, 81, color='yellow', alpha=0.3)
plt.axvspan(84, 86, color='yellow', alpha=0.3)
plt.axvspan(89, 91, color='yellow', alpha=0.3)
plt.axvspan(94, 96, color='yellow', alpha=0.3)


ax1.set_xlabel("Time Hours", fontsize=fontsize)

plt.text(3, 45, 'Action = 5 hrs', fontsize=fontsize)
plt.text(3, 100, 'Action = 10 hrs', fontsize=fontsize)

plt.savefig("Action_Atack_300.png", transparent=True)

#plt.show()

"""
"""
# Dispacthed Delivery Profile Ploting

export_profit= np.load("DQN_export_profit .npy", allow_pickle=True )
cost_of_islands= np.load("DQN_cost_of_islands.npy", allow_pickle=True )
battries_delivered= np.load("DQN_battries_delivered .npy", allow_pickle=True )
plt.figure(figsize=((12,8)), dpi=dpi)
ax1 = plt.subplot(4,1,1)

ax2 = ax1.twinx()
ax1.bar(np.arange(1,len(export_profit)+1,1), export_profit,width=0.5, bottom=0, align='edge' )
ax2.plot(np.arange(1 ,len(battries_delivered)+1,1),battries_delivered, "r*-")
ax1.bar(np.arange(1,len(cost_of_islands)+1,1), cost_of_islands,  width=0.5, bottom=0, align='center',)

ax1.set_xticks(np.arange(1,len(export_profit)+1,1))
ax2.set_yticks(np.arange(0,10,3))
ax2.set_ylim(0,10)
ax1.set_ylim(0,3000)
ax1.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)




export_profit= np.load("DDQN_export_profit .npy", allow_pickle=True )
cost_of_islands= np.load("DDQN_cost_of_islands.npy", allow_pickle=True )
battries_delivered= np.load("DDQN_battries_delivered .npy", allow_pickle=True )

ax1 = plt.subplot(4,1,2)
ax2 = ax1.twinx()

ax1.bar(np.arange(1,len(export_profit)+1,1), export_profit,width=0.5, bottom=0, align='edge' )
ax2.plot(np.arange(1,len(battries_delivered)+1,1),battries_delivered, "b*-")
ax1.bar(np.arange(1,len(cost_of_islands)+1,1), cost_of_islands,  width=0.5, bottom=0, align='center',)

ax1.set_xticks(np.arange(1,len(export_profit)+1,1))
ax2.set_yticks(np.arange(0,10,3))
ax2.set_ylim(0,10)
ax1.set_ylim(0,3000)
ax1.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)

ax1.legend(["BS","CS"], bbox_to_anchor=(-0.1, 1.0, 0.8, 2.7),loc="center", ncol=4, fontsize=15)
ax2.legend(["Shipped"], bbox_to_anchor=(0.0, 1.0, 1.140, 2.7),loc="center", ncol=4, fontsize=15)


export_profit= np.load("Dueling_DDQN_export_profit .npy", allow_pickle=True )
cost_of_islands= np.load("Dueling_DDQN_cost_of_islands.npy", allow_pickle=True )
battries_delivered= np.load("Dueling_DDQN_battries_delivered .npy", allow_pickle=True )

ax1 = plt.subplot(4,1,3)
ax2 = ax1.twinx()

ax1.bar(np.arange(1,len(export_profit)+1,1), export_profit,width=0.5, bottom=0, align='edge' )
ax2.plot(np.arange(1,len(battries_delivered)+1,1),battries_delivered, "b*-")
ax1.bar(np.arange(1,len(cost_of_islands)+1,1), cost_of_islands,  width=0.5, bottom=0, align='center',)

ax1.set_xticks(np.arange(1,len(export_profit)+1,1))
ax2.set_yticks(np.arange(0,len(battries_delivered),3))
ax2.set_ylim(0,10)
ax1.set_ylim(0,3000)
ax1.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)

ax2.set_ylabel("          No. of Batteries", loc = "bottom" , fontsize=fontsize)
ax1.set_ylabel("                Profit $", loc = "bottom" , fontsize=fontsize)

export_profit= np.load("Dueling_DQN_export_profit .npy", allow_pickle=True )
cost_of_islands= np.load("Dueling_DQN_cost_of_islands.npy", allow_pickle=True )
battries_delivered= np.load("Dueling_DQN_battries_delivered .npy", allow_pickle=True )

ax1 = plt.subplot(4,1,4)
ax2 = ax1.twinx()

ax1.bar(np.arange(1,len(export_profit)+1,1), export_profit,width=0.5, bottom=0, align='edge' )
ax2.plot(np.arange(1,len(battries_delivered)+1,1),battries_delivered, "b*-")
ax1.bar(np.arange(1,len(cost_of_islands)+1,1), cost_of_islands,  width=0.5, bottom=0, align='center',)

ax1.set_xticks(np.arange(1,len(export_profit)+1,1))
ax2.set_yticks(np.arange(0,len(battries_delivered),3))
ax2.set_ylim(0,10)
ax1.set_ylim(0,3000)
ax1.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)

ax1.set_xlabel("Dispatched Delivery (LIN)", fontsize=fontsize)
plt.text(0.4, 7, 'Dueling DQN', fontsize=fontsize)
plt.text(0.4, 19.5, 'Dueling DDQN', fontsize=fontsize)
plt.text(0.4, 30.5, 'DDQN', fontsize=fontsize)
plt.text(0.4, 44, 'DQN', fontsize=fontsize)
plt.savefig("Dispatched Delivery_300.png", transparent=True)


#plt.show()





"""


# Plot Functionality of All Algorithms
plt.figure(figsize=((12,8)), dpi=dpi)


bat_use = np.load("DQN_Load_Island_Bat_utilization.npy", allow_pickle=True )

bat_ship=np.load("DQN_Load_Island_Bat_at_ship.npy",  allow_pickle=True)

mt_load = np.load("DQN_Load_Island_MT.npy", allow_pickle=True)

deg_load =np.load("DQN_Load_Island_DEG.npy", allow_pickle=True) 
   
mt_time=np.load("DQN_Load_Island_MT_Time.npy",  allow_pickle=True)
deg_time =np.load("DQN_Load_Island_DEG_Time.npy", allow_pickle=True)

all_time =np.load("DQN_All_Time.npy", allow_pickle=True)
time_step = 168 #len(all_time)
start = 0
end = 96

print(len(all_time))
print(len(bat_use))

#fig, ax1 = plt.subplots(figsize=((10,3)), dpi=100)
ax1 = plt.subplot(4,1,1)
#ax1 = plt.subplot(2,2,1)
ax1.set_xticks([])

#plt.subplots(figsize=((10,3)), dpi=100)
ax2 = ax1.twinx()
ax1.bar(all_time[0:96], bat_use[0:96], color = "green")
ax1.plot(bat_ship, "b*-")
ax2.bar(mt_time, mt_load,  width=0.5, bottom=0, align='center',)
ax2.bar(deg_time, deg_load,  width=0.5, bottom=10, align='edge',)

ax1.set_ylim(0,10)
ax2.set_xlim(0,96)
#ax2.set_ylim(0,41)
ax2.set_yticks(np.arange(0,51, 25))
ax1.set_yticks(np.arange(0,11, 5))
ax1.tick_params(axis='y', labelsize=15)
#ax1.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)
#plt.xticks(np.arange(0,time_step+1,24))

ax1.legend([ "Batteries_at_ship", "LIN__Charged_Batteries", "MT", "DEG" ], bbox_to_anchor=(0.1, 1.0, 0.54, 0.4),loc="center", ncol=4, fontsize=15)
ax2.legend([ "MT", "DEG" ], bbox_to_anchor=(0.0, 1.0, 1.0, 0.4),loc="upper right", ncol=2, fontsize=15)



bat_use = np.load("DDQN_Load_Island_Bat_utilization.npy", allow_pickle=True )

bat_ship=np.load("DDQN_Load_Island_Bat_at_ship.npy",  allow_pickle=True)

mt_load = np.load("DDQN_Load_Island_MT.npy", allow_pickle=True)

deg_load =np.load("DDQN_Load_Island_DEG.npy", allow_pickle=True) 
   
mt_time=np.load("DDQN_Load_Island_MT_Time.npy",  allow_pickle=True)
deg_time =np.load("DDQN_Load_Island_DEG_Time.npy", allow_pickle=True)

all_time =np.load("DQN_All_Time.npy", allow_pickle=True)


#fig, ax1 = plt.subplots(figsize=((10,3)), dpi=100)
ax1 = plt.subplot(4,1,2)
#ax1 = plt.subplot(2,2,2)
#plt.subplots(figsize=((10,3)), dpi=100)
ax1.set_xticks([])
ax2 = ax1.twinx()
ax1.bar(all_time[0:96], bat_use[0:96], color = "green")
ax1.plot(bat_ship, "b*-")
ax2.bar(mt_time, mt_load,  width=0.5, bottom=0, align='center',)
ax2.bar(deg_time, deg_load,  width=0.5, bottom=10, align='edge',)


ax1.set_ylim(0,10)
ax2.set_xlim(0,end)
#ax2.set_ylim(0,41)
ax2.set_yticks(np.arange(0,51, 25))
ax1.set_yticks(np.arange(0,11, 5))
ax1.tick_params(axis='y', labelsize=15)
#ax1.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)
#ax2.set_ylim(0,35)




bat_use = np.load("D_DDQN_Load_Island_Bat_utilization.npy", allow_pickle=True )

bat_ship=np.load("D_DDQN_Load_Island_Bat_at_ship.npy",  allow_pickle=True)

mt_load = np.load("D_DDQN_Load_Island_MT.npy", allow_pickle=True)

deg_load =np.load("D_DDQN_Load_Island_DEG.npy", allow_pickle=True) 
   
mt_time=np.load("D_DDQN_Load_Island_MT_Time.npy",  allow_pickle=True)
deg_time =np.load("D_DDQN_Load_Island_DEG_Time.npy", allow_pickle=True)

all_time =np.load("DQN_All_Time.npy", allow_pickle=True)


#fig, ax1 = plt.subplots(figsize=((10,3)), dpi=100)
ax1 = plt.subplot(4,1,3)
#ax1 = plt.subplot(2,2,3)
#plt.subplots(figsize=((10,3)), dpi=100)
ax1.set_xticks([])
ax2 = ax1.twinx()
ax1.bar(all_time[0:96], bat_use[0:96], color = "green")
ax1.plot(bat_ship, "b*-")
ax2.bar(mt_time, mt_load,  width=0.5, bottom=0, align='center',)
ax2.bar(deg_time, deg_load,  width=0.5, bottom=10, align='edge',)


ax1.set_ylim(0,10)
ax2.set_xlim(0,end)
#ax2.set_ylim(0,41)
ax2.set_yticks(np.arange(0,51, 25))
ax1.set_yticks(np.arange(0,11, 5))
ax1.tick_params(axis='y', labelsize=15)
#ax1.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)
ax1.set_ylabel("      No. of Batteries", loc = "bottom" , fontsize=fontsize)
ax2.set_ylabel("             Power KW", loc = "bottom" , fontsize=fontsize)

#ax2.set_ylim(0,35)



bat_use = np.load("D_DQN_Load_Island_Bat_utilization.npy", allow_pickle=True )

bat_ship=np.load("D_DQN_Load_Island_Bat_at_ship.npy",  allow_pickle=True)

mt_load = np.load("D_DQN_Load_Island_MT.npy", allow_pickle=True)

deg_load =np.load("D_DQN_Load_Island_DEG.npy", allow_pickle=True) 
   
mt_time=np.load("D_DQN_Load_Island_MT_Time.npy",  allow_pickle=True)
deg_time =np.load("D_DQN_Load_Island_DEG_Time.npy", allow_pickle=True)

all_time =np.load("DQN_All_Time.npy", allow_pickle=True)


#fig, ax1 = plt.subplots(figsize=((10,3)), dpi=100)
ax1 = plt.subplot(4,1,4)
#ax1 = plt.subplot(2,2,4)
#plt.subplots(figsize=((10,3)), dpi=100)
ax2 = ax1.twinx()
ax1.bar(all_time[0:96], bat_use[0:96], color = "green")
ax1.plot(bat_ship, "b*-")
ax2.bar(mt_time, mt_load,  width=0.5, bottom=0, align='center',)
ax2.bar(deg_time, deg_load,  width=0.5, bottom=10, align='edge',)


ax1.set_ylim(0,10)
ax2.set_xlim(0,end)
#ax2.set_ylim(0,41)
ax2.set_yticks(np.arange(0,51, 25))
ax1.set_yticks(np.arange(0,11, 5))

ax1.set_xticks(np.arange(0,end+1,24))
ax1.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)

ax1.set_xlabel("Time Hours", fontsize=fontsize)
plt.text(3, 52, 'Dueling DQN', fontsize=fontsize)
plt.text(3, 112, 'Dueling DDQN', fontsize=fontsize)
plt.text(3, 172, 'DDQN', fontsize=fontsize)
plt.text(3, 222, 'DQN', fontsize=fontsize)
#plt.show()
plt.savefig("1_Functionality_Profile_300.png", transparent=True)
