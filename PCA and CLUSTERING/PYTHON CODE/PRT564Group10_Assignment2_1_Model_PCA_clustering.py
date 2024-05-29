################################################### Assignment 2 ###########################################################
# Unit: PRT564 - Data Analytics and Visualization                                                                          #
# Group Name: Group 10                                                                                                     #
#   Group member:                                                                                                          #
#       Anne (Dao Phuong Anh) Ta    - S359453                                                                              # 
#       Khai Quang Thang            - S367530                                                                              #   
#       Buu Dang Phan               - S373294                                                                              #
#       Van Phuc Vinh Ho            - S366270                                                                              #
############################################################################################################################
# Project objectives                                                                                                       #
# 1. To explore relevant, interesting, and actionable trends of past retractions (essential objective)                     #
# 2. To predict important aspects of future retractions (desirable objective)                                              #
############################################################################################################################
# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import kstest
import math
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode
from sklearn.metrics import silhouette_score
# Part 0: INPUT DATA
df = pd.read_csv("stagingdata_model_all_.csv")
#Model 1: Run data with only full text for subject, short-term subject will be excluded.
df_model1 = df.iloc[:,:158] 

#Model 2: Exclude column having full text subject
filtered_columns = df.filter(regex='^(?!.*\[Subject\]).*$') 
print(filtered_columns)
df_model2 = df[filtered_columns.columns]

# Part I: explore PCA components of the dataset
# Find the good number of component for each model based on 
# the threshold of PCA according to 90%, 95%, & 99%
def PCA_components(data, n_components=None):
    # Perform PCA
    pca = PCA(n_components=n_components)
    
    projected_data = pca.fit_transform(data)
    
    # compare the cumulative explained variance versus number of PCA components
    pca = PCA().fit(data)
    
    # visualise digits over 2 PCA components
    plt.scatter(projected_data[:, 0], projected_data[:, 1],
            c=range(len(projected_data)), edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', len(projected_data)))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()
    
    # Plot cumulative explained variance versus number of components
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()
    
    print("Without Standardisation")
    # Determine number of components required to explain the specified percentage of variance
    explained_variance_thresholds=[0.90, 0.95, 0.99]
    for threshold in explained_variance_thresholds:
        # Determine number of components required to explain the specified percentage of variance
        pca = PCA(threshold).fit(data)
        print("%.0f%% variance is explained by: %d components." % ((threshold * 100), pca.n_components_))
        
def PCA_components_standardisation(data, n_components=None):
    # Data preprocessing: Standardization
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)    
    projected_data = pca.fit_transform(df_scaled)
    
    # compare the cumulative explained variance versus number of PCA components
    pca = PCA().fit(df_scaled)
    
    # visualise digits over 2 PCA components
    plt.scatter(projected_data[:, 0], projected_data[:, 1],
            c=range(len(projected_data)), edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', len(projected_data)))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()
    
    # Plot cumulative explained variance versus number of components
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()
    
    print("With Standardisation")
    # Determine number of components required to explain the specified percentage of variance
    explained_variance_thresholds=[0.90, 0.95, 0.99]
    
    for threshold in explained_variance_thresholds:
        # Determine number of components required to explain the specified percentage of variance
        pca = PCA(threshold).fit(df_scaled)
        print("%.0f%% variance is explained by: %d components." % ((threshold * 100), pca.n_components_))
        
        
def PCA_components_ratioCheck(data, standardise=False, n_components=None):
    # Data preprocessing: Standardization
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(data)
    # Perform PCA
    pca = PCA(n_components=n_components)    
    if standardise==False:        
        # Calculate cumulative explained variance
        pca_data = pca.fit_transform(data)        
        print("explained ratio: ", pca.explained_variance_ratio_)
    else:
        # Calculate cumulative explained variance
        pca_data = pca.fit_transform(df_scaled)        
        print("explained ratio: ", pca.explained_variance_ratio_)
        
        
# print("PCA model 1")
# PCA_components_ratioCheck(df_model1,False,n_components=3)
# PCA_components_ratioCheck(df_model1,True, n_components=4)
PCA_components(df_model1,n_components=2)
PCA_components_standardisation(df_model1,n_components=2)
# Without standardisation
# PCA model 1: n_component = 3
# 90% variance is explained by: 2 components.
# 95% variance is explained by: 2 components.
# 99% variance is explained by: 3 components.

# With standardisation
# PCA model 1: n_component = 4
# 90% variance is explained by: 4 components.
# 95% variance is explained by: 4 components.
# 99% variance is explained by: 4 components.

print("PCA model 2")
# PCA_components_ratioCheck(df_model2,n_components=6)
PCA_components(df_model2,n_components=2)
PCA_components_standardisation(df_model2,n_components=2)
# Without standardisation
# PCA model 2: n_component = 6
# 90% variance is explained by: 2 components.
# 95% variance is explained by: 2 components.
# 99% variance is explained by: 6 components.
# With standardisation
# PCA model 2: n_component = 10
# 90% variance is explained by: 9 components.
# 95% variance is explained by: 10 components.
# 99% variance is explained by: 11 components.


# Part II: Create Dataset with PCA components, explored above
# Ojective: 
# Obj_1: Finallize the list exploratory variables have strong influence to extracted papers.
# Obj_2: Define the common function to 
def perform_pca(df, n_components):
    # Data preprocessing: Standardization
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
   # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_scaled)
    # print("Show Principle Components")
    # print(abs(pca.components_))
    
    # result_df = pd.DataFrame(data=abs(pca.components_))
    # result_df.columns = df.columns
    # file_name = str(n_components)+"PCA_result.csv"
    # result_df.to_csv(file_name, index=False)
    
    # Create a new DataFrame from the principal components
    principal_df = pd.DataFrame(data=principal_components, 
                                columns=[f'PC{i}' for i in range(1, n_components+1)])
    
    # Print explained variance ratio by each principal component
    pca_ns = PCA().fit(df)
    pca_s = PCA().fit(df_scaled)
    explained_variance_ratio = pca.explained_variance_ratio_
    print('Explained variation per principal component (no standardisation): {}'.format(pca_ns.explained_variance_ratio_))
    print('Explained variation per principal component (standardisation): {}'.format(pca_s.explained_variance_ratio_))
    for i, variance in enumerate(explained_variance_ratio):
        print(f"Principal Component {i+1}: {variance}")
    return principal_df
print('PCA performance of model 1')
PCA_perform_1 = perform_pca(df_model1,n_components=6)


print('PCA performance of model 2')
PCA_perform_2 = perform_pca(df_model2,n_components=11)

