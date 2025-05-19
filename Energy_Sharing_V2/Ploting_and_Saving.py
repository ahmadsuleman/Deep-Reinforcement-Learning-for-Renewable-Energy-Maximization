# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 00:16:34 2021

@author: Suleman_Sahib
"""

import numpy as np 
import matplotlib.pyplot as plt

from Modules import Battery_, WIND_, PV_, Critical_Load_, Non_Critical_Load_, Micro_Turbine_, DEG_
from Island_Generator import Island_
#==========================================================================
# Plot Generation Dataset
#############################
import pandas as pd

ti = 97
fsize = 15
"""
pv = np.load('PV_Actual.npy', allow_pickle=True) 
pvp = np.load('PV_Dayahead.npy', allow_pickle=True)
pv = 230 *( pv/max(pv) )
pvp = 230 * (pvp/max(pvp))
pv = pv[0:8760]
pvp = pvp[0:8760]
#pv = np.asarray(pv)
#pvp = np.asarray(pvp)
#acc =  (pv) - (pvp) #((pv - pvp)/pv)*100
#print(acc)
plt.plot(pv)
for i, data in enumerate(pv):
    if data == "nan":
        print(i)
        
print(pv.size)
plt.show()

l = np.load('Load_Dayahead.npy', allow_pickle=True)
lp = np.load('Load_Actual.npy', allow_pickle=True)
l = 15*( l/max(l))
lp = 15* (lp/max(lp))

l = l[0:8760]
lp = lp[0:8760]

w = np.load('Wind_Actual.npy', allow_pickle=True)
w = 200 *( w/max(w))
w = w[0:8760]
wp = np.load('Wind_Dayahead.npy', allow_pickle=True)
wp = 200*(wp/max(wp))

wp = wp[0:8760]




plt.figure(figsize=(10,7), dpi=100)

plt.subplot(3,1,1)
plt.plot(pv, 'bo-')
plt.plot(pvp, "r*--")
plt.legend(["PV Actual", "PV Predicted"], loc="upper left", ncol=2, fontsize=fsize)
#plt.xlabel("Time (hours)", fontsize=15)
plt.ylabel("Power (KW)", fontsize=fsize)
plt.xticks(np.arange(0, ti, step=24), color="black", fontsize=fsize)
plt.yticks(np.arange(0, 201, step=100), color="black", fontsize=fsize)
#plt.ylim(0,210)
#plt.xlim(0,ti-1)
plt.grid()

plt.subplot(3,1,2)
plt.plot(w, 'bo-')
plt.plot(wp, "r*--")
plt.legend(["Wind Actual", "Wind Predicted"], loc="lower right", ncol=2, fontsize=fsize)
#plt.xlabel("Time (hours)", fontsize=15)
plt.ylabel("Power (KW)", fontsize=fsize)
plt.xticks(np.arange(0, ti, step=24), color="black", fontsize=fsize)
plt.yticks(np.arange(0, 201, step=100), color="black", fontsize=fsize)
#plt.ylim(0,210)
#plt.xlim(0,ti-1)
plt.grid()

plt.subplot(3,1,3)
plt.plot(l, 'bo-')
plt.plot(lp, "r*--")
plt.legend(["Load Actual", "Load Predicted"], loc="lower right", ncol=2, fontsize=fsize)
plt.xlabel("Time (hours)", fontsize=fsize)
plt.ylabel("Power (KW)", fontsize=fsize)
#plt.xticks(np.arange(0, ti, step=24), color="black", fontsize=fsize)
#plt.yticks(np.arange(0, 21, step=10), color="black", fontsize=fsize)
#plt.ylim(0,20)
#plt.xlim(0,ti-1)
plt.grid()
plt.show()
#plt.savefig("E:/Energy_Sharing_V2/Results_Plots/source_profile_600.png", transparent=True)

"""

#==========================================================================================
# Island Simulations
#===================

# Source Island
pv   = PV_(capacity = 230)
wind = WIND_(capacity = 200)
c_load = Critical_Load_(capacity= 10)
nc_load = Non_Critical_Load_(capacity = 5)
env = Island_(PV=pv.pv_actual , WIND=wind.wind_actual, C_Load=c_load.critical_load_actual, NC_Load = nc_load.non_critical_load_actual, Total_Battries = 10)

# Source Load Island
pv   = PV_(capacity = 180)
wind = WIND_(capacity = 150)
c_load = Critical_Load_(capacity=30)
nc_load = Non_Critical_Load_(capacity = 20)
env1 = Island_(PV=pv.pv_actual , WIND=wind.wind_actual, C_Load=c_load.critical_load_actual, NC_Load = nc_load.non_critical_load_actual, Total_Battries = 10)


# Load Island          
pv   = PV_(capacity = 0)
wind = WIND_(capacity = 0)
c_load = Critical_Load_(capacity= 50)
nc_load = Non_Critical_Load_(capacity = 20)
env2 = Island_(PV=pv.pv_actual , WIND=wind.wind_actual, C_Load=c_load.critical_load_actual, NC_Load = nc_load.non_critical_load_actual, Total_Battries = 10)
env2.import_battries()


for t  in range(24):
    env.manage_energy(t)
    env1.manage_energy(t)
    env2.manage_energy(t)
    if env.export_status == 2:
        env.export_battries()
    if env2.import_status == 2:
        env2.import_battries(9)
    

fsize = 12
fsize1 = 15
st = 1
plt.figure(figsize=(20,10), dpi=300)
plt.subplot(3,2,1)
y = np.asarray(env.battries_charged)
x = np.asarray(env.all_time_steps)
mask1 = x < 14
mask2 = x > 14

plt.bar(x[mask1], y[mask1],  width=0.5, bottom=0, align='center',  color = 'green')
plt.bar(x[mask2], y[mask2],  width=0.5, bottom=0, align='center', color = 'red')
plt.legend(["Loacl Battries", "Swiped Battries"], bbox_to_anchor=(0.0, 1.0, 1.0, 0.2), loc="center", ncol=2)
plt.xticks(np.arange(0, 25, step=st), color="black", fontsize=fsize)
plt.yticks(np.arange(0, 11, step=2), color="black", fontsize=fsize)
plt.xlim(0,24)
plt.ylim(0,10)
plt.ylabel("No. of Bat (SI)", fontsize=fsize1)

#---------------------------------------------------------------------------
plt.subplot(3,2,2)

leg = []
col = ["#ff0000", "#800080" ,  "#ffff00" , "#ba8759",  "#ffa500",  "#f984ef" , "#560319",  "#a52a2a",  "#00ff00",  "#008000"]
for no, b in enumerate(env.battries):
    
    #print(len(b.soc_plot), len(b.time_step))

    #plt.subplot(2,5,(no+1))
    width = 0.5
    align = "edge"
    
    if no % 2==0:
        width = 0.5
        align = 'center'
        
    plt.bar(b.time_step, b.soc_plot,width = width,bottom=0, align = align,  color = col[no])
    leg.append(f"Bat # {no+1}")
    #plt.xlim(b.time_step[0], b.time_step[-1])
    #.xticklabels(b.time_step)
plt.legend(leg, bbox_to_anchor=(0.0, 1.0, 1.0, 0.2), loc="center", ncol=5) #
#plt.legend(leg,bbox_to_anchor=(1, 1, 0, 0), )
plt.xticks(np.arange(0, 25, step=st), color="black", fontsize=fsize)
plt.yticks(np.arange(0, 1.1, step=0.2), color="black", fontsize=fsize)
plt.xlim(0,24)
plt.ylim(0,1.1)
plt.ylabel("SOC (SI)", fontsize=fsize1)




#*****#######################################################################################
plt.subplot(3,2,3)

y = np.asarray(env1.battries_charged)
x = np.asarray(env1.all_time_steps)
mask1 = y < 9
mask2 = y >= 9

plt.bar(x[mask1], y[mask1],  width=0.5, bottom=0, align='center',  color = 'green')
plt.bar(x[mask2], y[mask2],  width=0.5, bottom=0, align='center', color = 'red')
#plt.legend(["Loacl Battries", "Swiped Battries"], bbox_to_anchor=(0.0, 1.0, 1.0, 0.2), loc="center", ncol=2, fontsize=fsize)
plt.xticks(np.arange(0, 25, step=st), color="black", fontsize=fsize)
plt.yticks(np.arange(0, 11, step=2), color="black", fontsize=fsize)
plt.xlim(0,24)
plt.ylim(0,10)
plt.ylabel("No. of Bat (SLI)", fontsize=fsize1)

#---------------------------------------------------------------------------
plt.subplot(3,2,4)

leg = []
col = ["#ff0000", "#800080" ,  "#ffff00" , "#ba8759",  "#ffa500",  "#f984ef" , "#560319",  "#a52a2a",  "#00ff00",  "#008000"]
for no, b in enumerate(env1.battries):
    
    #print(len(b.soc_plot), len(b.time_step))

    #plt.subplot(2,5,(no+1))
    width = 0.5
    align = "edge"
    
    if no % 2==0:
        width = 0.5
        align = 'center'
        
    plt.bar(b.time_step, b.soc_plot,width = width,bottom=0, align = align,  color = col[no])
    leg.append(f"Bat # {no+1}")
    #plt.xlim(b.time_step[0], b.time_step[-1])
    #.xticklabels(b.time_step)
#plt.legend(leg, bbox_to_anchor=(0.0, 1.0, 1.0, 0.2), ncol=5, fontsize=fsize) #
#plt.legend(leg,bbox_to_anchor=(1, 1, 0, 0), )
plt.xticks(np.arange(0, 25, step=st), color="black", fontsize=fsize)
plt.yticks(np.arange(0, 1.1, step=0.2), color="black", fontsize=fsize)
plt.xlim(0,24)
plt.ylim(0,1.1)
plt.ylabel("SOC (SLI)", fontsize=fsize1)

#*****++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plt.subplot(3,2,5)

y = np.asarray(env2.battries_charged)
x = np.asarray(env2.all_time_steps)
mask1 = x < 22
mask2 = x >= 22

plt.bar(x[mask1], y[mask1],  width=0.5, bottom=0, align='center',  color = 'green')
plt.bar(x[mask2], y[mask2],  width=0.5, bottom=0, align='center', color = 'red')
#plt.legend(["Loacl Battries", "Swiped Battries"], bbox_to_anchor=(0.0, 1.0, 1.0, 0.2), loc="center", ncol=2, fontsize=fsize)
plt.xticks(np.arange(0, 25, step=st), color="black", fontsize=fsize)
plt.yticks(np.arange(0, 11, step=2), color="black", fontsize=fsize)
plt.xlim(0,24)
plt.ylim(0,10)
plt.ylabel("No. of Bat (LIN)", fontsize=fsize1)
plt.xlabel("Time (hours)", fontsize=fsize1)

#---------------------------------------------------------------------------
plt.subplot(3,2,6)

leg = []
col = ["#ff0000", "#800080" ,  "#ffff00" , "#ba8759",  "#ffa500",  "#f984ef" , "#560319",  "#a52a2a",  "#00ff00",  "#008000"]
for no, b in enumerate(env2.battries):
    
    #print(len(b.soc_plot), len(b.time_step))

    #plt.subplot(2,5,(no+1))
    width = 0.5
    align = "edge"
    
    if no % 2==0:
        width = 0.5
        align = 'center'
        
    plt.bar(b.time_step, b.soc_plot,width = width,bottom=0, align = align,  color = col[no])
    leg.append(f"Bat # {no+1}")
    #plt.xlim(b.time_step[0], b.time_step[-1])
    #.xticklabels(b.time_step)
#plt.legend(leg, bbox_to_anchor=(0.0, 1.0, 1.0, 0.2), ncol=5, fontsize=fsize) #
#plt.legend(leg,bbox_to_anchor=(1, 1, 0, 0), )
plt.xticks(np.arange(0, 25, step=st), color="black", fontsize=fsize)
plt.yticks(np.arange(0, 1.1, step=0.2), color="black", fontsize=fsize)
plt.xlim(0,24)
plt.ylim(0,1.1)
plt.ylabel("SOC (LIN)", fontsize=fsize1)
plt.xlabel("Time (hours)", fontsize=fsize1)

plt.subplots_adjust(left=0.1,
                    bottom=0.2, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.2)


plt.savefig("E:/Energy_Sharing_V2/Results_Plots/BAT_Profile_300.png", transparent=True)
#plt.show()

########################333333################################################



# Future Prediction Ploting Sources

"""
plt.figure(figsize=(10,5))
#plt.bar(env.all_time_steps,env.possible_ship_times, width=0.5, bottom=0, align='center',color=(0.9,  0.4, 0.5))
y = np.asarray(env.possible_ship_times)
x = np.asarray(env.all_time_steps)
mask1 = y <= 1
mask2 = y > 1

plt.bar(x[mask1], y[mask1],  width=0.5, bottom=0, align='center',  color = 'red')
plt.bar(x[mask2], y[mask2],  width=0.5, bottom=0, align='center', color = 'green')


plt.xlim(0, 24)
#plt.ylim(0, 3)
plt.title("Possible Times for Ship", fontsize=20)
plt.xlabel("Time in Hours", fontsize=15)
#.ylabel("Yes = 1, No = 2", fontsize=15)
plt.xticks(np.arange(0, 24, step=1))
plt.yticks(np.asarray([1,2]), labels = ["No", "Yes"], fontsize=15)
  
plt.savefig("E:\Energy_Sharing_V2\Results_Plots\SHIP_T.png", transparent=True)
"""


"""
np.save("E:/Energy_Sharing_V2/Results_Data/battries_charged_at_timestep.npy", env.battries_charged_at_timestep, allow_pickle=True)
np.save("E:/Energy_Sharing_V2/Results_Data/critical_demand_timesteps.npy", env.critical_demand_timesteps, allow_pickle=True)
np.save("E:/Energy_Sharing_V2/Results_Data/non_critical_demand_timesteps.npy", env.non_critical_demand_timesteps, allow_pickle=True)
np.save("E:/Energy_Sharing_V2/Results_Data/battery_drain_timesteps.npy", env.battery_drain_timesteps, allow_pickle=True)
np.save("E:/Energy_Sharing_V2/Results_Data/non_re_source.npy", env.non_re_source, allow_pickle=True)
np.save("E:/Energy_Sharing_V2/Results_Data/possible_ship_times.npy", env.possible_ship_times, allow_pickle=True)
"""
"""
plt.figure(figsize=(10,5))    
plt.bar(env.all_time_steps, env.critical_demand_timesteps, width=0.5, bottom=0, align='center')
plt.bar(env.all_time_steps, env.non_critical_demand_timesteps, width=0.5, bottom=0, align='center')
plt.xlim(0, 24)
#plt.ylim(0, 3)
plt.title("Critical and Non Critical Demand", fontsize=20)
plt.xlabel("Time in Hours", fontsize=15)
plt.ylabel("Energy KW", fontsize=15)
#
plt.xticks(np.arange(0, 24, step=1))
plt.legend(["Critical Demand", "Non Critical Demand"], fontsize=15)
 
plt.savefig("E:\Energy_Sharing_V2\Results_Plots\Demand.png", transparent=True)

col_map = plt.get_cmap('Dark2')
plt.figure(figsize=(10,5))    
plt.bar(env.all_time_steps, env.battries_charged_at_timestep, width=0.5, bottom=0, align='center', color=col_map.colors, edgecolor='green' )
plt.xlim(0, 24)
#plt.ylim(0, 3)
plt.title("Fully Charged Battries", fontsize=20)
plt.xlabel("Time in Hours", fontsize=15)
plt.ylabel("Number of Battries", fontsize=15)
plt.xticks(np.arange(0, 24, step=1))
#plt.legend(["Number of Charged Battries"], fontsize=15)

plt.savefig("E:\Energy_Sharing_V2\Results_Plots\BAT_FULL.png", transparent=True)



plt.figure(figsize=(10,5))
#plt.bar(env.all_time_steps,env.possible_ship_times, width=0.5, bottom=0, align='center',color=(0.9,  0.4, 0.5))
y = np.asarray(env.possible_ship_times)
x = np.asarray(env.all_time_steps)
mask1 = y <= 1
mask2 = y > 1

plt.bar(x[mask1], y[mask1],  width=0.5, bottom=0, align='center',  color = 'red')
plt.bar(x[mask2], y[mask2],  width=0.5, bottom=0, align='center', color = 'green')


plt.xlim(0, 24)
#plt.ylim(0, 3)
plt.title("Possible Times for Ship", fontsize=20)
plt.xlabel("Time in Hours", fontsize=15)
#.ylabel("Yes = 1, No = 2", fontsize=15)
plt.xticks(np.arange(0, 24, step=1))
plt.yticks(np.asarray([1,2]), labels = ["No", "Yes"], fontsize=15)
  
plt.savefig("E:\Energy_Sharing_V2\Results_Plots\SHIP_T.png", transparent=True)


plt.figure(figsize=(10,5))    
plt.bar(env.all_time_steps, env.battery_drain_timesteps)
plt.xlim(0, 24)
#plt.ylim(0, 3)
plt.title("Battery Drain Time Steps", fontsize=20)
plt.xlabel("Time in Hours", fontsize=15)
plt.ylabel("Drain Time Steps", fontsize=15)
#
plt.xticks(np.arange(0, 24, step=1))
#plt.legend(["Battery Discharging Times"], fontsize=15)

plt.savefig("E:\Energy_Sharing_V2\Results_Plots\DRAIN.png", transparent=True)

col_map = plt.get_cmap('Dark2')
plt.figure(figsize=(10,5))
plt.bar(env.all_time_steps, env.mt_on_time, width=0.3, bottom=0, align='edge',  color='red')
plt.bar(env.all_time_steps, env.deg_on_time, width=0.2, bottom=0, align='center',  color='yellow')
plt.xlim(0, 24)
plt.legend(["MT_STATUS", "DEG_STATUS"], fontsize=15)
#plt.ylim(0, 3)
plt.title("DEG and Micro Turbine ON/OFF Status", fontsize=20)
plt.xlabel("Time in Hours", fontsize=15)
#.ylabel("Yes = 1, No = 2", fontsize=15)
plt.xticks(np.arange(0, 24, step=1))
plt.yticks(np.asarray([1,2]), labels = ["OFF", "ON"], fontsize=15)
plt.savefig("E:/Energy_Sharing_V2/Results_Plots/NON_RE_T.png", transparent=True)



#box_plot_data=[value1,value2,value3,value4]
plt.figure(figsize=(10,10))
plt.boxplot(env.charging_tracking,patch_artist=True, labels=["Bat 1", "Bat 2", "Bat 3", "Bat 4", "Bat 5", "Bat 6", "Bat 7", "Bat 8", "Bat 9","Bat 10"], )
plt.yticks(np.arange(0,24, step= 1))
plt.title("Individual Battery Charging with Time", fontsize=20)
#plt.xlabel(fontsize=15)
plt.ylabel("Time in Hours", fontsize=15)

plt.savefig("E:\Energy_Sharing_V2\Results_Plots\BAT_IND.png", transparent=True)
"""



# Future Prediction Ploting Load     
        
"""
plt.figure(figsize=(10,5))
#plt.bar(env.all_time_steps,env.possible_ship_times, width=0.5, bottom=0, align='center',color=(0.9,  0.4, 0.5))
y = np.asarray(env.remaining_battries_at_timestep)
x = np.asarray(env.all_time_steps)
mask1 = y <= 4
mask2 = y > 4
plt.bar(x[mask1], y[mask1],  width=0.5, bottom=0, align='center',  color = 'red')
plt.bar(x[mask2], y[mask2],  width=0.5, bottom=0, align='center', color = 'green')
plt.xlim(0, 24)
plt.ylim(0, 21)
plt.title("Remaining Fully Charged Battries [   LOAD ISLAND   ]", fontsize=20)
plt.xlabel("Time in Hours", fontsize=15)
plt.ylabel("Number of Battries", fontsize=15)
plt.xticks(np.arange(0, 24, step=1), color="black")
plt.yticks(np.arange(0, 20, step=1), color="black")
plt.savefig("E:/Energy_Sharing_V2/Results_Plots/BAT_Remaining_LOAD_ISLANDs.png", transparent=True)


plt.figure(figsize=(10,5))
#plt.bar(env.all_time_steps,env.possible_ship_times, width=0.5, bottom=0, align='center',color=(0.9,  0.4, 0.5))
y = np.asarray(env.battery_required_timesteps)
x = np.asarray(env.all_time_steps)
mask1 = y <= 1
mask2 = y > 1

plt.bar(x[mask1], y[mask1],  width=0.5, bottom=0, align='center',  color = 'green')
plt.bar(x[mask2], y[mask2],  width=0.5, bottom=0, align='center', color = 'red')

plt.xlim(0, 24)
#plt.ylim(0, 3)
plt.title("Ship Arrival Time [   LOAD ISLAND   ]", fontsize=20)
plt.xlabel("Time in Hours", fontsize=15)
#.ylabel("Yes = 1, No = 2", fontsize=15)
plt.xticks(np.arange(0, 24, step=1))
plt.yticks(np.asarray([1,2]), labels = ["No", "Yes"], fontsize=15)
plt.savefig("E:\Energy_Sharing_V2\Results_Plots\SHIP_T_LOAD_ISLANDs.png", transparent=True)

plt.figure(figsize=(10,5))
plt.bar(env.all_time_steps, env.mt_on_time, width=0.3, bottom=0, align='edge',  color='red')
plt.bar(env.all_time_steps, env.deg_on_time, width=0.2, bottom=0, align='center',  color='yellow')
plt.xlim(0, 24)
plt.legend(["MT_STATUS", "DEG_STATUS"], fontsize=15)
#plt.ylim(0, 3)
plt.title("DEG and Micro Turbine ON/OFF Status [   LOAD ISLAND   ]", fontsize=20)
plt.xlabel("Time in Hours", fontsize=15)
#.ylabel("Yes = 1, No = 2", fontsize=15)
plt.xticks(np.arange(0, 24, step=1))
plt.yticks(np.asarray([1,2]), labels = ["OFF", "ON"], fontsize=15)
plt.savefig("E:/Energy_Sharing_V2/Results_Plots/NON_RE_T_LOAD_ISLANDs.png", transparent=True)

np.save("E:/Energy_Sharing_V2/Results_Data/mt_on_time_LOAD_ISLANDs.npy", env.mt_on_time, allow_pickle=True)

np.save("E:/Energy_Sharing_V2/Results_Data/deg_on_time_LOAD_ISLANDs.npy", env.deg_on_time, allow_pickle=True)

np.save("E:/Energy_Sharing_V2/Results_Data/Ship_Arivals_LOAD_ISLANDs.npy", env.battery_required_timesteps, allow_pickle=True)

np.save("E:/Energy_Sharing_V2/Results_Data/remaining_battries_at_timestep_LOAD_ISLANDs.npy", env.remaining_battries_at_timestep, allow_pickle=True)

"""
##############################################################################
"""
# Plot Generation Dataset
ti = 97
fsize = 15
pv = dataset['forecast solar day ahead'][0:ti]
pvp = dataset['generation solar'][0:ti]
pv = pv/max(pv)
pvp = pvp/max(pvp)


l = dataset['total load forecast'][0:ti]
lp = dataset['total load actual'][0:ti]
l = l/max(l)
lp = lp/max(lp)

w = dataset['forecast wind onshore day ahead'][0:ti]
w = w/max(w)
wp = dataset['generation wind onshore'][0:ti]
wp = wp/max(wp)

plt.figure(figsize=(10,7), dpi=600)

plt.subplot(3,1,1)
plt.plot(pv, 'bo-')
plt.plot(pvp, "r*--")
plt.legend(["PV Actual", "PV Predicted"], loc="upper left", ncol=2, fontsize=fsize)
#plt.xlabel("Time (hours)", fontsize=15)
plt.ylabel("Power (KW)", fontsize=fsize)
plt.xticks(np.arange(0, ti, step=24), color="black", fontsize=fsize)
plt.yticks(np.arange(0, 1.2, step=0.5), color="black", fontsize=fsize)
plt.ylim(0,1.09)
plt.xlim(0,ti-1)
plt.grid()

plt.subplot(3,1,2)
plt.plot(w, 'bo-')
plt.plot(wp, "r*--")
plt.legend(["Wind Actual", "Wind Predicted"], loc="lower right", ncol=2, fontsize=fsize)
#plt.xlabel("Time (hours)", fontsize=15)
plt.ylabel("Power (KW)", fontsize=fsize)
plt.xticks(np.arange(0, ti, step=24), color="black", fontsize=fsize)
plt.yticks(np.arange(0, 1.2, step=0.5), color="black", fontsize=fsize)
plt.ylim(0,1.09)
plt.xlim(0,ti-1)
plt.grid()

plt.subplot(3,1,3)
plt.plot(l, 'bo-')
plt.plot(lp, "r*--")
plt.legend(["Load Actual", "Load Predicted"], loc="lower right", ncol=2, fontsize=fsize)
plt.xlabel("Time (hours)", fontsize=fsize)
plt.ylabel("Power (KW)", fontsize=fsize)
plt.xticks(np.arange(0, ti, step=24), color="black", fontsize=fsize)
plt.yticks(np.arange(0, 1.2, step=0.5), color="black", fontsize=fsize)
plt.ylim(0,1.09)
plt.xlim(0,ti-1)
plt.grid()
plt.savefig("E:/Energy_Sharing_V2/Results_Plots/source_profile_tfs_600.png", transparent=True)
"""
