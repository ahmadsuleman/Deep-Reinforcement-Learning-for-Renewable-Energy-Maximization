# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:43:08 2021

@author: Suleman_Sahib
"""


import tensorflow as tf
import torch
import numpy as np
import gym 
from gym import Env, spaces

from Modules import WIND_, PV_, Critical_Load_, Non_Critical_Load_
from Island_Generator import Island_

# Source Island
pv   = PV_(capacity = 230)#115 ) #92) #184 ) #
wind = WIND_(capacity = 200) # 100 ) #80) #160) #
c_load = Critical_Load_(capacity= 10)
nc_load = Non_Critical_Load_(capacity = 5)
SI = Island_(PV=pv.pv_actual , WIND=wind.wind_actual, C_Load=c_load.critical_load_actual, NC_Load = nc_load.non_critical_load_actual, Total_Battries = 10)

# Source Load Island
pv   = PV_(capacity = 180)# 90 ) #72) #144) #
wind = WIND_(capacity =  150)# 75) #60) #120) #
c_load = Critical_Load_(capacity=30)
nc_load = Non_Critical_Load_(capacity = 20)
SLI = Island_(PV=pv.pv_actual , WIND=wind.wind_actual, C_Load=c_load.critical_load_actual, NC_Load = nc_load.non_critical_load_actual, Total_Battries = 10)


# Load Island          
pv   = PV_(capacity = 0)
wind = WIND_(capacity = 0)
c_load = Critical_Load_(capacity= 50)
nc_load = Non_Critical_Load_(capacity = 20)
LIN = Island_(PV=pv.pv_actual , WIND=wind.wind_actual, C_Load=c_load.critical_load_actual, NC_Load = nc_load.non_critical_load_actual, Total_Battries = 10)




# This grid settings are high dimension and difficult to render.
# Required Grid Settings
Grid_Rows= 3
Grid_Col = 4


SOURCE_ISLAND = (0, 0)
SOURCE_LOAD = (2, 0)
LOAD_ISLAND = (1,3)
MAX_CAP = 9

RES_1 = (0,3)
RES_2 = (2,3)
RES_3 = (0,2)
RES_4 = (2,2)

class ship_movement:
    
    def __init__(self):
        self.source_island = SI
        self.source_load   = SLI
        self.load_island   = LIN
       
        self.bat_at_ship = 0
        self.time_step = 0
      
        self.print_grid = np.zeros([Grid_Rows, Grid_Col])
        self.print_grid[SOURCE_ISLAND] = 9
        self.print_grid[SOURCE_LOAD] = 2
        self.print_grid[LOAD_ISLAND] = 4
        self.state = SOURCE_ISLAND
        self.state_space = []
        self.action_space = spaces.Discrete(5)
        self.next_state = []
        self.game_steps = 0
        self.simulate_islands()
        
    
    def move_ship(self,state, action):
        
        if action == 0:
                next_position = (state[0] - 1, state[1]) # up
        elif action == 1:
                next_position = (state[0] + 1, state[1]) # down
        elif action == 2:
                next_position = (state[0], state[1] - 1) # left
        elif action == 3:
                next_position = (state[0], state[1]) # stay
        else:
                next_position = (state[0], state[1] + 1) # right
        
        
        # Keep the next state within specified range. 
        if (next_position[0] >= 0) and (next_position[0] <= (Grid_Rows -1)):
            if (next_position[1] >= 0) and (next_position[1] <= (Grid_Col -1)):
                    return next_position 
                    
        return state 
       
                
    
    def bat_loader(self,next_loc, action):
        reward = 0
        done = False
        self.game_steps += 1
        
        # Move to the destination with shortest route. 
        
        if ((next_loc != SOURCE_ISLAND) and (next_loc != SOURCE_LOAD) and (next_loc != LOAD_ISLAND) and (next_loc != (1,0)) and (next_loc != (1,1)) and (next_loc != (1,2))):
            reward += -1
           
        #if ((next_loc == (1,0)) and (next_loc == (1,2)) and (next_loc == (1,3))and (self.bat_at_ship == 9)):
         #  reward += 1
            
            
    
        if (next_loc == LOAD_ISLAND) and (self.bat_at_ship == 0):
            reward += -0.5
        
        #if (next_loc == LOAD_ISLAND) and (self.bat_at_ship == 9):
        #    reward += 1.5
           
        if ( (next_loc == LOAD_ISLAND) and (self.bat_at_ship == 9)):
            self.load_island.import_battries(self.bat_at_ship)
            self.bat_at_ship = 0
            self.game_steps = 0
            done = True
            reward += 1000
            return reward , done
        
        if self.game_steps > 23:
            reward = -1000
            self.bat_at_ship = 0
            self.game_steps = 0
            done = True
            return reward , done
            
            
           
        if ((next_loc == SOURCE_LOAD) or (next_loc == SOURCE_ISLAND) ) and (self.bat_at_ship >= 9):
            reward += -0.7
            pass
            
        if (next_loc == SOURCE_ISLAND) and (self.bat_at_ship == 0):
            self.bat_at_ship = self.source_island.export_battries()
            reward += 1
            
        if (next_loc == SOURCE_LOAD) and (self.bat_at_ship == 0):
            self.bat_at_ship = self.source_load.export_battries()
            reward += 1
        
        if (next_loc == SOURCE_ISLAND) and ((self.bat_at_ship > 0) and (self.bat_at_ship < 9)):
            self.bat_at_ship += self.source_island.export_battries()
            reward += 0.5
        
        if (next_loc == SOURCE_LOAD) and ((self.bat_at_ship > 0) and (self.bat_at_ship < 9)):
            self.bat_at_ship += self.source_load.export_battries()
            reward += 0.5

            
        
        if (self.bat_at_ship > 9):
            self.bat_at_ship = 0
            

       
        return reward    , done
    
    def step(self,action):
        reward = 0
        
        done = False
        
        
        next_loc = self.move_ship(self.state, action)
        reward, done = self.bat_loader(next_loc, action)
        self.simulate_islands() 
        
        
        
        #self.next_state = []
        #self.next_state = list(next_loc)
        #self.next_state.append(self.bat_at_ship)
        #charged_battries =[self.source_island.battries_charged[-1], self.source_load.battries_charged[-1], self.load_island.battries_charged[-1]  ]
        #self.next_state.extend(charged_battries)
        
        self.state = next_loc
        s = []
        s.extend(self.state)
        s.append(self.bat_at_ship)
        s.append(self.game_steps)
        #s.append(self.load_island.battries_charged[-1])
        
        return torch.Tensor(s), reward, done
         
   
    
    def simulate_islands(self):
            self.source_island.manage_energy(self.time_step)
            self.source_load.manage_energy(self.time_step)
            self.load_island.manage_energy(self.time_step)
            self.time_step += 1
            if self.time_step > 8759:
                self.time_step = 0
    
    def non_re_pen(self):
            all_pen = []
            reward = 0
            
            p_load_mt = round(self.load_island.mt_.bill)
            all_pen.append(p_load_mt)
            
            p_load_deg = round(self.load_island.deg_.bill)
            all_pen.append(p_load_deg)
            
            p_source_mt = round(self.source_island.mt_.bill)
            all_pen.append(p_source_mt)
            
            
            p_source_deg = round(self.source_island.deg_.bill)
            all_pen.append(p_source_deg)
            
            p_source_load_mt = round(self.source_load.mt_.bill)
            all_pen.append(p_source_load_mt)
            
            p_source_load_deg = round(self.source_load.deg_.bill)
            all_pen.append(p_source_load_deg)
            
            self.source_load.deg_.reset()
            self.source_load.mt_.reset()
            self.source_island.deg_.reset()
            self.source_island.mt_.reset()
            self.load_island.deg_.reset()
            self.load_island.mt_.reset()
            
          
            
            non_re = sum(all_pen)
            
            
            
            
            
            
            
            return -non_re
    
    def reset(self):
        s = []
        self.state = LOAD_ISLAND # self.state
        self.bat_at_ship = 0 #self.bat_at_ship
        self.game_steps = 0
        #self.simulate_islands()
        s.extend(self.state)
        s.append(self.bat_at_ship)
        s.append(self.game_steps)
        #s.append(self.load_island.battries_charged[-1])
        
        #all_state.append(np.asarray(self.bat_at_ship))
        #all_state.append(np.asarray(self.state))
        #return np.asarray(all_state)
        #self.next_state = []
        #self.next_state = list(self.state)
        #self.next_state.append(self.bat_at_ship)
        #charged_battries =[self.source_island.battries_charged[-1], self.source_load.battries_charged[-1], self.load_island.battries_charged[-1]  ]
        #self.next_state.extend(charged_battries)
        #np.asarray(s)
        return torch.Tensor(s)
        
        
    def render_grid(self):
        self.print_grid[self.state] = 1
        for i in range(0, Grid_Rows):
            print("---------------------------------")
            out = '| '
            for j in range(0, Grid_Col):
                if self.print_grid[i, j] == 1:
                    token = f'SHIP({self.bat_at_ship})'
                if self.print_grid[i, j] == 2:
                    token = '  SLI  '
                if self.print_grid[i, j] == 0:
                    token = '       '
                if self.print_grid[i, j] == 9:
                    token = '  SI   '
                if self.print_grid[i, j] == 4:
                    token = '  LIN  '
                out += token + '|'
            print(out)
        print("---------------------------------")
        
        self.print_grid[self.state] = 0
        self.print_grid[SOURCE_ISLAND] = 9
        self.print_grid[SOURCE_LOAD] = 2
        self.print_grid[LOAD_ISLAND] = 4
        
"""
env = ship_movement()
num_actions = env.action_space.n
# Test Section
class DQN(tf.keras.Model):
  #Dense neural network class
  def __init__(self):
    super(DQN, self).__init__()
    self.dense1 = tf.keras.layers.Dense(32, activation="relu")
    self.dense2 = tf.keras.layers.Dense(num_actions, dtype=tf.float32) # No activation
    
  def call(self, x):
    #Forward pass

    x = self.dense1(x)
    
    return self.dense2(x)

main_nn = DQN()
target_nn = DQN()

optimizer = tf.keras.optimizers.Adam(1e-4)
mse = tf.keras.losses.MeanSquaredError()

state = env.reset()

print(state.shape)
print(np.array(state))
state = tf.expand_dims(state, axis=0)
#q_vlaue = main_nn(state)
#print(tf.reduce_max(q_vlaue))



for t in range(1000):
    action = env.action_space.sample()
    
    next_state, reward, done = env.step(action)
    #next_state[].insert(self.bat_at_ship)
    env.render_grid()
    #print(env.bat_at_ship, reward)
    

    print(next_state, reward, env.bat_at_ship, action, t)
    if done:
        break

"""