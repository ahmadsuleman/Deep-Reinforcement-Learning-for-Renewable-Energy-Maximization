# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 01:09:27 2022

@author: Suleman_Sahib
"""

import numpy as np

import matplotlib.pyplot as plt

price_data = [0.35,0.35,0.35,0.35,0.35,0.35,0.45,0.45,0.71,0.71,0.71,0.71,0.45,0.45,0.45,0.45,0.45,0.71,0.71,0.71,0.71,0.71,0.35,0.35]
price_dataset = []

for t in range(365):
    price_dataset.extend(price_data)

plt.figure(figsize=(10,5))
print(len(price_data))   
plt.plot(price_data)
plt.xticks(np.arange(0,24,1))
plt.yticks(np.arange(0,1,0.1))
plt.ylim(0,1)
plt.grid()
plt.show() 
price_data = np.load("price_dataset.npy", allow_pickle=True)
#print(len(price_data))
for t in range(8760):
    print(price_data[t])
#np.save("price_dataset.npy",price_dataset, allow_pickle=True)