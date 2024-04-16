# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 20:31:10 2022

@author: Gianluca Pellecchia
"""
  
import pandas as pd
import numpy as np

regions = ["reg1", "reg2"]
resources = ["Solar", "Wind", "Hydro"]

years_list = []
for i in range(0,31):
    years_list.append("Y"+str(i))

index_day = []    
for day in range(0,365):
    index_day.append("day"+str(day))
    
hours = []
for hour in range(0,24):
    hours.append("H"+str(hour))
    
timesteps = []
for n in range(1,25):
    timesteps.append(n)

#%%
demand_regional = {}
for reg in regions:
    yearly_demand = pd.read_excel("Input_demand_hourly.xlsx", sheet_name= reg)    
    demand_yearly = {}
    for year in years_list:
        yearly = pd.DataFrame(yearly_demand[year])
        demand_yearly[year] = pd.DataFrame(np.reshape(yearly.values, (365,24)),index = index_day, columns = hours)
    demand_regional[reg] = demand_yearly

regional_averages = {}
for reg in regions:
    yearly_stack = pd.DataFrame()
    for year in years_list:
        hourly_average = []
        for hour in hours:
            avg = demand_regional[reg][year][hour].mean()
            hourly_average.append(avg)
        yearly_average = pd.DataFrame(np.vstack(hourly_average), columns = ["Demand"], index = pd.MultiIndex.from_product([[year], timesteps],names=["Years", "Timesteps"]))
        yearly_stack = pd.concat([yearly_stack,yearly_average])
    regional_averages[reg] = yearly_stack 

with pd.ExcelWriter('Average_demand.xlsx', engine="openpyxl", mode="w") as writer:
    regional_averages["reg1"]["Demand"].to_excel(writer, sheet_name="reg1")
for reg in regions: 
    if reg=="reg1":
        continue
    with pd.ExcelWriter('Average_demand.xlsx', engine="openpyxl", mode="a") as writer:
        regional_averages[reg].to_excel(writer, sheet_name=str(reg))

#%%
RES_regional = {}
for reg in regions:
    yearly_RES = pd.read_excel("Input_RES_hourly.xlsx", sheet_name= reg)   
    RES_resource = {}
    for res in resources:
        yearly_res = pd.DataFrame(yearly_RES[res])
        RES_resource[res] = pd.DataFrame(np.reshape(yearly_res.values,(365,24)), index = index_day, columns = hours)
    RES_regional[reg] = RES_resource 
    
regional_averages = {}
for reg in regions:
    yearly_res = {}
    for resource in resources:
        yearly_stack = pd.DataFrame()
        for year in years_list:
            hourly_average = []
            for hour in hours:
                avg = RES_regional[reg][resource][hour].mean()
                hourly_average.append(avg)
            yearly_average = pd.DataFrame(np.vstack(hourly_average), columns = [resource], index = pd.MultiIndex.from_product([[year], timesteps],names=["Years", "Timesteps"]))
            yearly_stack = pd.concat([yearly_stack,yearly_average])
        yearly_res[resource] = yearly_stack
    regional_averages[reg] = yearly_res 
    
with pd.ExcelWriter('Average_resource.xlsx', engine="openpyxl", mode="w") as writer:
    regional_averages["reg1"]["Solar"].to_excel(writer, sheet_name="reg1"+"-"+"Solar")
for reg in regions: 
    for res in resources:
        if reg=="reg1" and res=="Solar":
            continue
        with pd.ExcelWriter('Average_resource.xlsx', engine="openpyxl", mode="a") as writer:
            regional_averages[reg][res].to_excel(writer, sheet_name=str(reg)+"-"+str(res))

        
        
        