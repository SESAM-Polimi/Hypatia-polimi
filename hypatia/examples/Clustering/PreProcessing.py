# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:49:55 2023

@author: GP
"""
#%% import modules

import pandas as pd 
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import silhouette_score
from kneed import KneeLocator

#%%
# Starting from input demand as hourly demand with years on columns
regions = ["reg1", "reg2", "reg3", "reg4"]

Hypatia_format = {}
for reg in regions:
    yearly_demand = pd.read_excel("Input_demand_hourly.xlsx", sheet_name= reg) 
    num_rows, num_cols = yearly_demand.shape
    years = num_cols
    demand_hourly = []
    for i in range(0,years):
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
resources = ["Solar", "Wind", "Hydro"]

Hypatia_format_RES = {}
for reg in regions:
    yearly_RES = [pd.read_excel("Input_RES_hourly.xlsx", sheet_name= reg)] 
    repeated = yearly_RES*years
    yearly_RES = np.vstack(repeated)
    Hypatia_format_RES[reg] = pd.DataFrame(yearly_RES,columns=resources)
    
with pd.ExcelWriter('Hypatia_RES_CF.xlsx') as writer:
    Hypatia_format_RES["reg1"].to_excel(writer, sheet_name="reg1")
    Hypatia_format_RES["reg2"].to_excel(writer, sheet_name="reg2")
    Hypatia_format_RES["reg3"].to_excel(writer, sheet_name="reg3")
    Hypatia_format_RES["reg4"].to_excel(writer, sheet_name="reg4")

#%% utilities

regions = ["reg1", "reg2", "reg3", "reg4"]
resources = ["Solar", "Wind", "Hydro"]
items = ["Demand", "Solar", "Wind", "Hydro"]

years_list = []
for i in range(0,11):
    years_list.append("Y"+str(i))

index_day = []    
for day in range(0,365):
    index_day.append("day"+str(day))
    
hours = []
for hour in range(0,24):
    hours.append("H"+str(hour))
     
#%% Starting from input demand as hourly demand with years on columns    
    
demand_regional = {}
for reg in regions:
    yearly_demand = pd.read_excel("Input_demand_hourly.xlsx", sheet_name= reg)    
    demand_yearly = {}
    for year in years_list:
        yearly = pd.DataFrame(yearly_demand[year])
        demand_yearly[year] = pd.DataFrame(np.reshape(yearly.values, (365,24)),index = index_day, columns = hours)
    demand_regional[reg] = demand_yearly

#%% Starting from input RES capacity factor as hourly format, repeat per n years

RES_regional = {}
for reg in regions:
    yearly_RES = pd.read_excel("Input_RES_hourly.xlsx", sheet_name= reg)   
    RES_resource = {}
    for res in resources:
        yearly_res = pd.DataFrame(yearly_RES[res])
        RES_resource[res] = pd.DataFrame(np.reshape(yearly_res.values,(365,24)), index = index_day, columns = hours)
    RES_regional[reg] = RES_resource    
    
#%% Merge dataframe

input_clustering = {}
for reg in regions:
    total_yearly = {}
    for year in years_list:
        total = demand_regional[reg][year]
        for res in resources:
            total = total.join(RES_regional[reg][res], rsuffix="-"+res)
        total_yearly[year] = total
    input_clustering[reg] = total_yearly

#%% evaluating optimal number of cluster with dtw method and elbow method

k_rng = range(2,14)
SSE = []
a = input_clustering["reg1"]["Y0"]
aa = a.to_numpy()
b = aa[:,:,newaxis]
norm_hourly_demand = TimeSeriesScalerMeanVariance().fit_transform(b)
for k in k_rng:   
    model = TimeSeriesKMeans(n_clusters=k, 
                             n_init=2, 
                             metric="dtw", 
                             verbose=False, 
                             max_iter_barycenter=10,
                             random_state=0)                                              
    model.fit(norm_hourly_demand) 
    SSE.append(model.inertia_)
    
plt.figure()
plt.plot(k_rng,SSE)
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")

kl = KneeLocator(k_rng, SSE, curve="convex", direction="decreasing")
kSSE = kl.elbow

#%% starting clustering
# kSSE = 13


df_total = input_clustering
regional_means = {}
for reg in regions:
    yearly_means = {}
    for year in years_list:
        df = df_total[reg][year]
        narray = (df.to_numpy())[:,:,newaxis]
        # peak = pd.DataFrame(df.max(axis=0)).T
        # peak.rename(index={0: "Peak day"}, inplace = True)
        # peak.insert(0,"Cluster",kSSE)
        seed = 0
        np.random.seed(seed)
        norm_hourly_demand = TimeSeriesScalerMeanVariance().fit_transform(narray)
        model1 = TimeSeriesKMeans(n_clusters=kSSE, 
                                  n_init=2, 
                                  metric="dtw", 
                                  verbose=False, 
                                  max_iter_barycenter=10,
                                  random_state=seed)
        model1.fit(norm_hourly_demand)
        y_pred = model1.fit_predict(norm_hourly_demand)
        print("DBA silhoutte: {:.2f}".format(silhouette_score(norm_hourly_demand, y_pred, metric="dtw")))
        
        cluster_centers1 = pd.DataFrame()
        plt.figure()
        for yi in range(kSSE):
            plt.subplot(3, kSSE , yi+1)
            cluster_centers1['Cluster'+str(yi)] = model1.cluster_centers_[yi].ravel()
            for xx in model1._X_fit[y_pred == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.plot(model1.cluster_centers_[yi].ravel(), "r-")       
        
        df.insert(0,"Cluster",y_pred)
        df.head()
        
        # df = [df,peak]
        # df = pd.concat(df)
         
        clusters = {}
        for i in range(kSSE):
            clusters['Cluster '+str(i+1)] = df[df.Cluster==i].loc[:,df.columns != 'Cluster']
        
        items_cluster = {}   
        count=0
        for g in range(0,len(items)*24, 24):
                cluster_sectors = {}
                for i in range(kSSE):
                    cluster_sectors['Cluster '+str(i+1)] = df[df.Cluster==i].loc[:,df.columns != 'Cluster'].iloc[:, g:g+24]
                items_cluster[items[count]] = cluster_sectors
                count+=1
    
        x=range(1,25)
        items_means = {}
        for item in items:
            cluster_mean = {}
            for xx, yy in items_cluster[item].items():  
                plt.figure()
                plt.plot(x,yy.T, "k-", alpha=.2)
                plt.plot(x,yy.T.mean(axis=1), "r-")
                plt.xlabel('Hours')
                plt.ylabel('Load demand')
                plt.title(str(item)+ '-' + str(xx)+ '-' + str(year))
                cluster_mean['Mean '+str(xx)] = yy.T.mean(axis=1)
            items_means[item] = cluster_mean
            
        yearly_means[year] = items_means
    regional_means[reg] = yearly_means


#%% rearrangement

x=range(1,25)
items_means_reg = {}
for reg in regions:
    items_means_yearly = {}
    for item in items:
        item_yearly_mean_cluster = {}
        for year in years_list:
            items_mean_cluster = {}
            for xx, yy in regional_means[reg][year][item].items():  
                items_mean_cluster[xx] = pd.DataFrame(yy)
            item_yearly_mean_cluster [year] = items_mean_cluster
        items_means_yearly[item] = item_yearly_mean_cluster
    items_means_reg[reg] = items_means_yearly
    
#%% print to excel

clusters_list = []
for i in range(0, kSSE):
    clusters_list.append("Mean Cluster "+str(i+1) + " - ")
    
timesteps = []
for n in range(1,24*(kSSE)+1):
    timesteps.append(n)
    
excel_reg = {}
for reg in regions:
    excel_items = {}
    for item in items:
        item_stack = pd.DataFrame()
        for year in years_list:
            cluster_list = [] 
            for key in items_means_reg[reg][item][year].keys():
                cluster_list.append(items_means_reg[reg][item][year][key].values)
            cluster_stack = pd.DataFrame(np.vstack(cluster_list), columns = [item], index = pd.MultiIndex.from_product([[year], timesteps],names=["Years", "Timesteps"]))
            item_stack= pd.concat([item_stack,cluster_stack])
        excel_items[item] = item_stack
    excel_reg[reg] = excel_items
    
with pd.ExcelWriter('Clusters.xlsx', engine="openpyxl", mode="w") as writer:
    excel_reg["reg1"]["Demand"].to_excel(writer, sheet_name="reg1"+"-"+"Demand")
for reg in regions: 
    for item in items:
        if reg=="reg1" and item=="Demand":
            continue
        with pd.ExcelWriter('Clusters.xlsx', engine="openpyxl", mode="a") as writer:
            excel_reg[reg][item].to_excel(writer, sheet_name=str(reg)+"-"+str(item))
