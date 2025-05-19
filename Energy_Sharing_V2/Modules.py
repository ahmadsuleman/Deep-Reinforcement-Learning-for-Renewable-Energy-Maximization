# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 14:12:47 2021

@author: Suleman_Sahib
"""
import numpy as np

class Battery_:
    def __init__(self, size = 100, eta = 0.99999):
        self.size = size 
        self.eta  = eta
        self.soc  =  0
        self.state = 0
        self.status = 0
        self.soc_plot = []
        self.time_step = []
        self.curtail_amount = []
        self.load_curtail = 0
        self.curtail_ = 0
        self.soc_plot.append(self.soc)
        self.time_step.append(0)
        self.actually_stored = 0
        
    def reset(self):
        self.soc  =  0
        self.state = 0
        self.status = 0
        self.load_curtail = 0
        self.curtail_ = 0
        #self.soc_plot = []
        #self.time_step = []
        self.curtail_amount = []

    def charge(self, energy, t):
         # Charging
        self.curtail_ = 0
        if (self.soc < 1) and (self.status != 1) :
            self.state = 1
            storable = 1 -  self.soc
            required_for_storage = storable * self.size/ self.eta
            required_for_storage = round(required_for_storage,3)
            if (required_for_storage > energy):
                self.soc += (energy/ self.size) * self.eta
                self.soc = round(self.soc, 3)
                self.actually_stored = energy
                self.curtail_ = 0
            else:
                self.soc += round(required_for_storage/ self.size * self.eta , 3)
                self.soc = round(self.soc, 3)
                self.state = 0
                self.status = 1
                self.curtail_ = 0
                self.curtail_ = round (energy - required_for_storage, 3)
                self.actually_stored = required_for_storage
        else:
            self.state = 0
            #self.status = 1
        #self.curtail_amount.append(curtail_)
        self.soc_plot.append(self.soc)
        self.time_step.append(t)
        
    def drain(self, load, t):
        self.load_curtail = 0
        if self.state == 1:
            print("... Battery is under charging ...")
        elif (self.soc > 0.2) or (self.status == 1):
            self.state = 2
            dischargable = self.soc - 0.2
            dischargable = round(dischargable, 3)
            #print(f"storable energy {storable}")
            required_from_battery = round (load/self.size/ self.eta, 3)
            if (required_from_battery <= dischargable):
                self.soc -= required_from_battery 
            else:
                self.load_curtail =  required_from_battery - dischargable
                self.soc -= dischargable
                self.state = 0
                self.status = 0
        else:
            print("... Battery is empty ...")
            
        self.soc_plot.append(self.soc)
        self.time_step.append(t)        
        
class EV_:
    def __init__(self, size, eta):
        self.size = size 
        self.eta  = eta
        self.soc  =  0
        self.state = 0
        self.status = 0
        self.soc_plot = []
        self.time_step = []
        self.curtail_amount = []
        self.soc_plot.append(self.soc)
        
    def charge(self, energy, t):
         # Charging
        curtail_ = 0
        if self.state == 2:
            print("... Battery is  Discharging ...")
        elif (self.soc < 1) and (self.status != 1) :
            self.state = 1
            storable = 1 -  self.soc
            required_for_storage = storable * self.size/ self.eta
            required_for_storage = round(required_for_storage,3)
            if (required_for_storage > energy):
                self.soc += (energy/ self.size) * self.eta
                self.soc = round(self.soc, 3)
            else:
                self.soc += round(required_for_storage/ self.size * self.eta , 3)
                self.soc = round(self.soc, 3)
                self.state = 0
                self.status = 1
                curtail_ = round (energy - required_for_storage, 3)
        else:
            self.state = 0
            #self.status = 1
        self.curtail_amount.append(curtail_)
        self.soc_plot.append(self.soc)
        self.time_step.append(t)
        
    def drain(self, load, t):
        curtail_ = 0
        if self.state == 1:
            print("... Battery is under charging ...")
        elif (self.soc > 0.2) or (self.status == 1):
            self.state = 2
            dischargable = self.soc - 0.2
            dischargable = round(dischargable, 3)
            #print(f"storable energy {storable}")
            required_from_battery = round (load/self.size/ self.eta, 3)
            if (required_from_battery <= dischargable):
                self.soc -= required_from_battery 
            else:
                load_curtail =  required_from_battery - dischargable
                self.soc -= dischargable
                self.state = 0
                self.status = 0
        else:
            print("... Battery is empty ...")
            
        self.soc_plot.append(self.soc) 
        self.time_step.append(t)

class Micro_Turbine_:
    def __init__(self, capacity = 10, cost = 0.0253, ):
        self.generation_ = capacity
        self.cost = cost
        self.state = 0
        self.extra = 0
        self.bill = 0
        self.time_step = []
        self.gen_plt = []
        self.cost_plt = []
        self.time_step.append(0)
        self.gen_plt.append(self.generation_)
        self.cost_plt.append(self.bill)
        
    def turn_on(self, load, t):
        
        
        self.bill += self.cost * self.generation_
        self.state = 1
        if (load > self.generation_):
            self.extra = load - self.generation_
        else:
            self.extra = 0
        
        self.time_step.append(t)
        self.gen_plt.append(self.generation_)
        self.cost_plt.append(self.bill)
            
    def turn_off(self, t):
        self.state = 0
        self.generation_ = 0
        self.time_step.append(t)
        self.gen_plt.append(self.generation_)
        self.cost_plt.append(self.bill)
        
    def reset(self):
        self.bill = 0        
            
        
        
        
class DEG_:
    def __init__(self, cost = 0.431 ):
        # Functionality
        self.generation_ = 0
        self.cost = cost
        self.state = 0
        self.bill = 0
        self.time_step = []
        self.gen_plt = []
        self.cost_plt = []
        self.time_step.append(0)
        self.gen_plt.append(self.generation_)
        self.cost_plt.append(self.bill)
        
        
    def turn_on(self, load, t):
        self.generation_ = load
        self.bill += self.cost * self.generation_ 
        self.state = 1
        self.time_step.append(t)
        self.gen_plt.append(self.generation_)
        self.cost_plt.append(self.bill)
    
    def turn_off(self, t):
        self.state = 0
        self.generation_ = 0 
        self.time_step.append(t)
        self.gen_plt.append(self.generation_)
        self.cost_plt.append(self.bill)
        
    def reset(self):
        self.bill = 0
        
        
        
        

class PV_:
    def __init__(self, capacity =1):
        energy = np.load('PV_Actual.npy', allow_pickle=True)
        self.pv_actual = energy/max(energy)*capacity # maximum 100 MW
        
        energy = np.load('PV_Dayahead.npy', allow_pickle=True)
        self.pv_pred = energy/max(energy)*capacity # maximum 100 MW 
        
    def gen(self, time_step):
        return np.asarray([self.pv_actual[time_step], self.pv_pred[time_step]])
        
class WIND_:
    def __init__(self,capacity=1):        

        energy = np.load('Wind_Actual.npy', allow_pickle=True)
        self.wind_actual = energy/max(energy)*capacity# Maximum 50 MW
        
        energy = np.load('Wind_Dayahead.npy', allow_pickle=True)
        self.wind_pred = energy/max(energy)*capacity# Maximum 50 MW
    def gen(self, time_step):
        return np.asarray([self.wind_actual[time_step], self.wind_pred[time_step]])
        
        
        
                
class Critical_Load_:
    def __init__(self, capacity = 1):
        energy = np.load('Load_Dayahead.npy', allow_pickle=True)
        self.critical_load_pred = energy/max(energy)*capacity # maximum 1 MW

        energy = np.load('Load_Actual.npy', allow_pickle=True)
        self.critical_load_actual = energy/max(energy)*capacity # maximum 1 MW
    def load(self, time_step):
        return np.asarray([self.critical_load_actual[time_step], self.critical_load_dayahead[time_step]])
    
        
class Non_Critical_Load_:
    def __init__(self,capacity = 1):
        energy = np.load('Load_Dayahead.npy', allow_pickle=True)
        self.non_critical_load_pred = (energy/max(energy))*capacity

        energy = np.load('Load_Actual.npy', allow_pickle=True)
        self.non_critical_load_actual = (energy/max(energy))*capacity
        
    def load(self, time_step):
        return np.asarray([self.non_critical_load_actual[time_step], self.non_critical_load_dayahead[time_step]])
    
    
        
    

    


            
                