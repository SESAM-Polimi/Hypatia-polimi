import pandas as pd 
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import silhouette_score
from kneed import KneeLocator

regions = ["reg1", "reg2", "reg3", "reg4"]

years_list = []
for i in range(0,11):
    years_list.append("Y"+str(i))

index_day = []    
for day in range(0,365):
    index_day.append("day"+str(day))
    
hours = []
for hour in range(0,24):
    hours.append("H"+str(hour))
    
demand_regional = {}
for reg in regions:
    yearly_demand = pd.read_excel("Input_demand_hourly.xlsx", sheet_name= reg)    
    demand_yearly = {}
    for year in years_list:
        yearly = pd.DataFrame(yearly_demand[year])
        demand_yearly[year] = pd.DataFrame(np.reshape(yearly.values, (365,24)),index = index_day, columns = hours)
    demand_regional[reg] = demand_yearly
    
k_rng = range(2,14)
SSE = []
a = demand_regional["reg1"]["Y0"]
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

df_total = demand_regional
regional_means = {}
for reg in regions:
    yearly_means = {}
    for year in years_list:
        df = df_total[reg][year]
        narray = (df.to_numpy())[:,:,newaxis]
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
         
        clusters = {}
        for i in range(kSSE):
            clusters['Cluster '+str(i+1)] = df[df.Cluster==i].loc[:,df.columns != 'Cluster']
    
        x=range(1,25)
        cluster_mean = {}
        for xx, yy in clusters.items():  
            plt.figure()
            plt.plot(x,yy.T, "k-", alpha=.2)
            plt.plot(x,yy.T.mean(axis=1), "r-")
            plt.xlabel('Hours')
            plt.ylabel('Load demand')
            plt.title(str(xx)+ '-' + str(year))
            cluster_mean['Mean '+str(xx)] = yy.T.mean(axis=1)
            
        yearly_means[year] = cluster_mean
    regional_means[reg] = yearly_means
    
#%% print to excel

clusters_list = []
for i in range(0, kSSE):
    clusters_list.append("Mean Cluster "+str(i+1))
    
timesteps = []
for n in range(1,24*(kSSE)+1):
    timesteps.append(n)
    
excel_reg = {}
for reg in regions:
    item_stack = pd.DataFrame()
    for year in years_list:
        cluster_list = [] 
        for key in regional_means[reg][year].keys():
            cluster_list.append(regional_means[reg][year][key].values)
        cluster_stack = pd.DataFrame(np.hstack(cluster_list), columns = ["Demand"], index = pd.MultiIndex.from_product([[year], timesteps],names=["Years", "Timesteps"]))
        item_stack= pd.concat([item_stack,cluster_stack])
    excel_reg[reg] = item_stack
    
with pd.ExcelWriter('Clusters_demand.xlsx', engine="openpyxl", mode="w") as writer:
    excel_reg["reg1"]["Demand"].to_excel(writer, sheet_name="reg1")
for reg in regions: 
    if reg=="reg1":
        continue
    with pd.ExcelWriter('Clusters_demand.xlsx', engine="openpyxl", mode="a") as writer:
        excel_reg[reg].to_excel(writer, sheet_name=str(reg))
