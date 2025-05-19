# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 15:57:22 2021

@author: Suleman_Sahib
"""
from Modules import Battery_, WIND_, PV_, Critical_Load_, Non_Critical_Load_, Micro_Turbine_, DEG_
import numpy as np
import matplotlib.pyplot as plt


          
class Island_:
    def __init__(self,PV, WIND, C_Load, NC_Load, Total_Battries):
        self.battery_sharing_timesteps = []
        self.battery_drain_timesteps = []
        self.battery_required_timesteps = []
        self.generation_curtail_timesteps = []
        self.load_curtail_timesteps = []
        self.battery_full_timesteps = []
        self.critical_demand_timesteps = []
        self.non_critical_demand_timesteps = []
        self.energy_utilized = []
        
        self.all_time_steps = []
        self.charging_tracking = []
        self.possible_ship_times = []
        self.mt_on_time = []
        self.deg_on_time = []
        
        self.energy = []
        self.pv = PV
        self.wind = WIND
        self.c_load = C_Load
        self.nc_load = NC_Load
        self.total_load = C_Load + NC_Load
        self.total_bat = Total_Battries
        
        self.bat_socs = []
        self.waste = []
        self.battries_charged = []
        self.all_bat_status = []
        self.all_time_steps = []
        
        self.bat_no = 0
        self.drain_no = 0
        self.charge_count = 0
        self.sharing_status = 1
        self.import_cost = 0
        self.current_re_gen = 0
        self.actually_saved = 0
        
        
        self.bat_no = 0
        self.battries = [Battery_(size=100) for _  in range(Total_Battries)]
        self.mt_ = Micro_Turbine_()
        self.deg_ = DEG_()
        
        
    
    def import_battries(self, bat_imported):
            if self.charge_count == 0:
                self.charge_count = bat_imported
                #self.import_cost = 
                self.bat_no = 0
                self.drain_no = 0
                
                for ind, b in enumerate(self.battries):
                    if ind < self.charge_count:
                        b.soc = 1.0
                        b.status = 1
                    else:
                        b.reset()
                
                
    
    def export_battries(self):
        
        # (self.charge_count == (self.total_bat-1)) and (self.battries[self.charge_count].state != 1) :
            bat_exported = self.charge_count
            self.charge_count = 0
            self.bat_no = 0
            self.drain_no = 0
            for b in self.battries:
                b.reset()
            
            return bat_exported
            
            
    def save_in_battery(self, energy, t):
        #if self.battries[self.bat_no].status != 1: 
            
            self.battries[self.bat_no].charge(energy, t)
            self.actually_saved += self.battries[self.bat_no].actually_stored
            #self.actually_saved += self.battries[self.bat_no].curtail_
            if self.battries[self.bat_no].status == 1:
               
                ex = self.battries[self.bat_no].curtail_
                self.actually_saved += ex
                if (self.bat_no < (self.total_bat - 1)):
                    self.bat_no += 1
                    self.charge_count += 1
                    
                    self.battries[self.bat_no].charge(ex, t)
                    #self.actually_saved += ex
                    
                    ex = 0
            else:
                self.actually_saved += self.battries[self.bat_no].curtail_
            
    def non_re_source(self, load, t):
        self.mt_.turn_on(load, t)
        if self.mt_.extra > 0:
            self.deg_.turn_on(self.mt_.extra, t)
            self.mt_.turn_off
            self.deg_.turn_off
        else:
            self.mt_.turn_off
            
        self.mt_.turn_off
        self.deg_.turn_off
        
    def drain_from_battery(self, energy, t):
        #if (self.battries[self.drain_no].status == 0) and(self.drain_no < (self.total_bat - 1)) :
        if self.charge_count > 0:
            
            if (self.battries[self.drain_no].status == 0):
                self.charge_count -= 1
                for no, b in enumerate(self.battries):    
                    if b.status == 1:
                        self.drain_no = no
                        break
            if (self.battries[self.drain_no].soc > 0.2):
                self.battries[self.drain_no].drain(energy, t)
           # else:
           #     for no, b in enumerate(self.battries):    
           #         if b.status == 1:
           #             self.drain_no = no
           #             break
                
                
            #print("here inside", self.drain_no)
        else:
            if (self.pv[t] + self.wind[t]) == 0:
                if (self.battries[self.drain_no].soc > 0.2):
                    self.battries[self.drain_no].drain(energy, t)
                else:
                    self.non_re_source(energy, t)
            else:
                
                #print("No Battery Available Going for NON RE Sources", energy)
                self.non_re_source(energy, t)
            
        
    def manage_energy(self, time_step):
        t = time_step
        self.mt_.turn_off(time_step)
        self.deg_.turn_off(time_step)
        gen = self.pv[t] + self.wind[t]
        self.current_re_gen = gen
        load = self.c_load[t] + self.nc_load[t]
        energy = gen - load
        self.energy= [energy, self.pv[t], self.wind[t], self.c_load[t], self.nc_load[t]]
        #print(energy, t)
        if (gen > load):
                self.save_in_battery(energy, t)
                #util = 
                self.energy_utilized = load + self.actually_saved
                self.actually_saved = 0
        if (gen < load):
            if (gen > self.c_load[t]):
                self.energy_utilized = gen + load
                gen = gen - self.c_load[t]
            else:
                #print("Critical Load From Battery", t)
                self.drain_from_battery((self.c_load[t] - gen), t)
                gen = 0
            #print("NON Critical Load from battery", t)
            self.drain_from_battery((self.nc_load[t] - gen), t)
        
        
            
            
            
        
                 
        self.battries_charged.append(self.charge_count)            
        self.all_time_steps.append(t)
        
        
       
                   
            
        
        
    






    
