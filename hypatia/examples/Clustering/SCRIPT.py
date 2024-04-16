# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 11:55:02 2022

@author: Gianluca Pellecchia
"""

import pandas as pd
import numpy as np

#%%
# Starting from input demand as hourly demand with years on columns
regions = ["reg1", "reg2", "reg3", "reg4"]

Hypatia_format = {}
for reg in regions:
    yearly_demand = pd.read_excel("Input_demand_hourly.xlsx", sheet_name= reg) 
    num_rows, num_cols = yearly_demand.shape
    demand_hourly = []
    for i in range(0,num_cols):
        demand_hourly.append(yearly_demand.iloc[:,i])
    total_demand = np.hstack(demand_hourly)
    Hypatia_format[reg] = pd.DataFrame(total_demand)
    
with pd.ExcelWriter('Hypatia_demand.xlsx') as writer:
    Hypatia_format["reg1"].to_excel(writer, sheet_name="reg1")
    Hypatia_format["reg2"].to_excel(writer, sheet_name="reg2")
    Hypatia_format["reg3"].to_excel(writer, sheet_name="reg3")
    Hypatia_format["reg4"].to_excel(writer, sheet_name="reg4")

#%%
# Starting from input RES capacity factor as hourly format, repeat per n years
regions = ["reg1", "reg2", "reg3", "reg4"]

Hypatia_format = {}
for reg in regions:
    yearly_RES = [pd.read_excel("Input_RES_hourly.xlsx", sheet_name= reg)] 
    repeated = yearly_RES*11
    yearly_RES = np.vstack(repeated)
    # for i in range(0,num_cols):
    #     demand_hourly.append(yearly_demand.iloc[:,i])
    # total_demand = np.hstack(demand_hourly)
    # yearly_RES = pd.DataFrame(np.repeat(yearly_RES[0], 11, axis=0))
    Hypatia_format[reg] = pd.DataFrame(yearly_RES)
    
with pd.ExcelWriter('Hypatia_RES_CF.xlsx') as writer:
    Hypatia_format["reg1"].to_excel(writer, sheet_name="reg1")
    Hypatia_format["reg2"].to_excel(writer, sheet_name="reg2")
    Hypatia_format["reg3"].to_excel(writer, sheet_name="reg3")
    Hypatia_format["reg4"].to_excel(writer, sheet_name="reg4")
    
#%%
# Starting from input demand as total yearly

yearly_demand = pd.read_excel("Input_demand.xlsx", sheet_name="Sheet1") 
time_series = pd.read_excel("Time_series_demand.xlsx", sheet_name="Sheet1")

num_rows, num_cols = yearly_demand.shape
num_years = num_rows

demand = []
for i in range(0,num_rows):
    a = [yearly_demand.iloc[i]]
    demand_hourly = np.repeat([yearly_demand.iloc[i]], 8760, axis=0)
    demand.append(np.multiply(demand_hourly,time_series))
demand_total = np.vstack(demand)

# demand_hourly = np.zeros((8760,num_cols))
# demand = []
# for i in range(0,num_rows):
#     for j in range(0,8760):
#         demand_hourly[j,:] = yearly_demand.iloc[i]
#     demand.append(np.multiply(demand_hourly,time_series))
# demand_total = np.vstack(demand)

#%%
# To calculate the daily avarage

num_rows, num_cols = demand_total.shape

# avg = np.zeros((24*num_years,6))
average_dem= []
for h in range(0,num_rows,8760):
    demand_year = demand_total[h:(h+8760),:]
    for r in range(0,24,1):
        d = np.zeros((1,num_cols))
        for i in range(r,8760,24):
            d += demand_year[i,:]
        avg = d/365
        average_dem.append(avg)
average_demand = np.vstack(average_dem)
        
        
df = pd.DataFrame(average_demand)
df1 = df.set_axis(['Export', 'Primary', 'Transport', 'Industry', 'Residential', 'Services'], axis=1, inplace=False)
df1.to_excel('Average_demand.xlsx')

#%%
# Calculate the daily avarage of the resource capacity factors

pd.read_excel("Input_RES.xlsx", sheet_name="Sheet1") 

#salvare i dati dentro una variabile
xlsx = pd.read_excel("Input_RES.xlsx", sheet_name="Sheet1") 
# a=xlsx.iloc[8759][125]

avg = np.zeros((24,3))
d = 0

for j in range(0,3,1):
    for r in range (0,24,1):
        for i in range(r,8760,24):
            d = d + xlsx.iloc[i][j]
        avg[r,j] = d/365
        d = 0
        
df1 = pd.DataFrame(avg)
df1.to_excel('Average_RES.xlsx')

#%%

carrier_in = pd.read_excel("Input_carrier.xlsx", sheet_name="Sheet1")
n_rows, n_cols = carrier_in.shape

carrier_ratio = np.repeat(carrier_in.values, 24, axis=0)

dff = pd.DataFrame(carrier_ratio)
dff.to_excel('Carrier_ratio_in.xlsx')




