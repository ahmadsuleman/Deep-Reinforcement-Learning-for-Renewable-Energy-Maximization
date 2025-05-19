# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 10:27:37 2021

@author: Suleman_Sahib
"""
import math
epsilon= 1.0
epsilon_decay_1 = 0.000005
epsilon_decay_2 = 0.0000372
for episode in range(70000):
  if episode < 50000:
        epsilon -= epsilon_decay_1
        
  if (episode > 50000) and (episode < 70000):
        epsilon -= epsilon_decay_2
  print(epsilon)