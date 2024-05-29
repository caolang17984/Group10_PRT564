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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import itertools
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('df_model3.csv')
# Data preprocessing: Standardization
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
def PCA_components1(data, n_components=None):
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
def PCA_components(data, n_components=None):
    # Perform PCA
    pca = PCA(n_components=n_components)
    
    projected_data = pca.fit_transform(data)
    
    # Calculate cumulative explained variance
    pca = PCA().fit(data)
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
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
    # Determine number of components required to explain the specified percentage of variance
    explained_variance_thresholds=[0.90, 0.95, 0.99]
    for threshold in explained_variance_thresholds:
        # Determine number of components required to explain the specified percentage of variance
        pca = PCA(threshold).fit(data)
        print("%.0f%% variance is explained by: %d components." % ((threshold * 100), pca.n_components_))
        # print("Number of components required to explain %.0f%% of the variance:" % (explained_variance_threshold * 100), n_components_threshold)
    return cumulative_explained_variance

print('PCA component without Standardization')
PCA_result = PCA_components1(df,n_components=2)
print('PCA component with Standardization')
PCA_result = PCA_components1(df_scaled,n_components=2)



def visualize_pca(df, n_components):
    
    X = df.iloc[:,1:].values
    y = df.iloc[:,0].values
    # Create training and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # Apply Standardisation on explanatory variables in training set
    std_scaler = preprocessing.StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)

    # Initialise 2-component PCA
    pca = PCA(n_components=n_components)

    # PCA on original explanatory variables
    X_train_pca = pca.fit_transform(X_train)

    # PCA on standardised explanatory variables
    X_train_std_pca = pca.fit_transform(X_train_std)

    # Initialise a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,4))

    # Subplot 1 (original) containing scatterplots for class labels of wine
    for l,c,m in zip(range(1,4), ('blue', 'red', 'green'), ('^', 's', 'o')):
        ax1.scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train==l, 1],
            color=c,
            label='class %s' %l,
            alpha=0.5,
            marker=m
            )

    # Subplot 2 (standardised) containing scatterplots for class labels of wine
    for l,c,m in zip(range(1,4), ('blue', 'red', 'green'), ('^', 's', 'o')):
        ax2.scatter(X_train_std_pca[y_train==l, 0], X_train_std_pca[y_train==l, 1],
            color=c,
            label='class %s' %l,
            alpha=0.5,
            marker=m
            )
    
    # Set titles
    ax1.set_title('Original training dataset after PCA')    
    ax2.set_title('Standardised training dataset after PCA')    

    # Other settings
    for ax in (ax1, ax2):
        ax.set_xlabel('1st principal component')
        ax.set_ylabel('2nd principal component')
        ax.legend(loc='upper right')
        ax.grid()

    plt.tight_layout()
    plt.show()
print('PCA component without Standardization with 4 components')
PCA_visual = visualize_pca(df,3)


def perform_pca(df, n_components, column_names):  
   # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df)
    print("Show Principle Components")
    # print(abs(pca.components_))
    result_df = pd.DataFrame(data=abs(pca.components_))
    result_df.columns = column_names
    file_name = str(n_components)+"PCA_result.csv"
    result_df.to_csv(file_name, index=False)
    
    # Create a new DataFrame from the principal components
    principal_df = pd.DataFrame(data=principal_components, 
                                columns=[f'PC{i}' for i in range(1, n_components+1)])
    
    # Print explained variance ratio by each principal component
    explained_variance_ratio = pca.explained_variance_ratio_
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    for i, variance in enumerate(explained_variance_ratio):
        print(f"Principal Component {i+1}: {variance}")
    return principal_df
print('PCA component without Standardization without 3 components')
PCA_perform_1 = perform_pca(df,n_components=3, column_names=df.columns)
print('PCA component without Standardization with 8 components')
PCA_perform_2 = perform_pca(df_scaled,n_components=8, column_names=df.columns)




