import pandas as pd 
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import silhouette_score
from kneed import KneeLocator

regions = ["reg1", "reg2", "reg3", "reg4"]
resources = ["Solar", "Wind"] #, "Hydro"]

years_list = []
for i in range(0,11):
    years_list.append("Y"+str(i))

index_day = []    
for day in range(0,365):
    index_day.append("day"+str(day))
    
hours = []
for hour in range(0,24):
    hours.append("H"+str(hour))
    
RES_regional = {}
for reg in regions:
    yearly_RES = pd.read_excel("Input_RES_hourly.xlsx", sheet_name= reg)   
    RES_resource = {}
    for res in resources:
        yearly_res = pd.DataFrame(yearly_RES[res])
        RES_resource[res] = pd.DataFrame(np.reshape(yearly_res.values,(365,24)), index = index_day, columns = hours)
    RES_regional[reg] = RES_resource 

kSSE = {}
k_rng = range(2,14)
SSE = []
a = RES_regional["reg1"]["Solar"]
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
kSSE["Solar"]=kl.elbow

SSE = []
a = RES_regional["reg1"]["Wind"]
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
kSSE["Wind"]=kl.elbow

# SSE = []
# a = RES_regional["reg2"]["Hydro"]
# aa = a.to_numpy()
# b = aa[:,:,newaxis]
# norm_hourly_demand = TimeSeriesScalerMeanVariance().fit_transform(b)
# for k in k_rng:   
#     model = TimeSeriesKMeans(n_clusters=k, 
#                              n_init=2, 
#                              metric="dtw", 
#                              verbose=False, 
#                              max_iter_barycenter=10,
#                              random_state=0)                                              
#     model.fit(norm_hourly_demand) 
#     SSE.append(model.inertia_)
    
# plt.figure()
# plt.plot(k_rng,SSE)
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")

# kl = KneeLocator(k_rng, SSE, curve="convex", direction="decreasing")
# kSSE["Hydro"]=kl.elbow

# if kSSE["Hydro"]==None:
#     kSSE["Hydro"]=kSSE["Solar"]

df_total = RES_regional
regional_means = {}
for reg in regions:
    resource_means = {}
    for res in resources:  
            
        df = df_total[reg][res]
        
        narray = (df.to_numpy())[:,:,newaxis]
        seed = 0
        np.random.seed(seed)
        norm_hourly_demand = TimeSeriesScalerMeanVariance().fit_transform(narray)
        model1 = TimeSeriesKMeans(n_clusters=kSSE[res], 
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
        for yi in range(kSSE[res]):
            plt.subplot(3, kSSE[res] , yi+1)
            cluster_centers1['Cluster'+str(yi)] = model1.cluster_centers_[yi].ravel()
            for xx in model1._X_fit[y_pred == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.plot(model1.cluster_centers_[yi].ravel(), "r-")       
        
        df.insert(0,"Cluster",y_pred)
        df.head()
         
        clusters = {}
        for i in range(kSSE[res]):
            clusters['Cluster '+str(i+1)] = df[df.Cluster==i].loc[:,df.columns != 'Cluster']
    
        x=range(1,25)
        cluster_mean = {}
        for xx, yy in clusters.items():  
            plt.figure()
            plt.plot(x,yy.T, "k-", alpha=.2)
            plt.plot(x,yy.T.mean(axis=1), "r-")
            plt.xlabel('Hours')
            plt.ylabel(res)
            plt.title(str(xx))
            cluster_mean['Mean '+str(xx)] = yy.T.mean(axis=1)
            
        resource_means[res] = cluster_mean
    regional_means[reg] = resource_means
    
#%% print to excel

# clusters_list = []
# for i in range(0, kSSE["Solar"]):
#     clusters_list.append("Mean Cluster "+str(i+1))
    
# timesteps = []
# for n in range(1,24*(kSSE["Solar"])+1):
#     timesteps.append(n)
    
excel_reg = {}
for reg in regions:
    cluster_stack = {}
    for res in resources:
        clusters_list = []
        timesteps = []
        for i in range(0, kSSE[res]):
            clusters_list.append("Mean Cluster "+str(i+1))
        for n in range(1,24*(kSSE[res])+1):
            timesteps.append(n)
        cluster_list = [] 
        for key in regional_means[reg][res].keys():
            cluster_list.append(regional_means[reg][res][key].values)
        cluster_stack[res] = pd.DataFrame(np.hstack(cluster_list), columns = [res], index = pd.MultiIndex.from_product([["Y0"], timesteps],names=["Years", "Timesteps"]))
        # item_stack[res]= pd.concat([item_stack,cluster_stack])
    excel_reg[reg] = cluster_stack
    
with pd.ExcelWriter('Clusters_resource.xlsx', engine="openpyxl", mode="w") as writer:
    excel_reg["reg1"]["Solar"].to_excel(writer, sheet_name="reg1"+"-"+"Solar")
for reg in regions: 
    for res in resources:
        if reg=="reg1" and res=="Solar":
            continue
        with pd.ExcelWriter('Clusters_resource.xlsx', engine="openpyxl", mode="a") as writer:
            excel_reg[reg][res].to_excel(writer, sheet_name=str(reg)+"-"+str(res))

