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

df = pd.read_csv('df_model3.csv')
# df_PCA.to_csv('PCA_new_dataset_model_df.csv', index=False)
# df_k = df.iloc[:,1::]
# apply standardisation
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
# apply pca
pca = PCA(n_components=2)
pca_df = scaler.fit_transform(df)
pca_df = pd.DataFrame(pca_df, columns=df.columns)
# apply pca standardisation 
pca = PCA(n_components=2)
pca_df_std = scaler.fit_transform(df)
pca_df_std = pd.DataFrame(pca_df_std, columns=df.columns)
# Function to find the optimal value of Kmean:
def find_optimal_k(df, max_k=10):
    df_k = df.iloc[:,1::]
    # Initialize list to store Elbrow scores
    inertia_scores = []
    # Initialize list to store silhouette scores
    sil = []
    kmeans_params = range(1, max_k + 1)
    kmeans_params_sil = range(2, max_k + 1)
    # Perform Elbow Method analysis
    for k in kmeans_params:
        model = KMeans(n_clusters=k, n_init=10)
        model.fit(df_k)
        inertia_scores.append(model.inertia_)
        
    # Perform Silhouette scores Method analysis to find the K value
    for k in kmeans_params_sil:
        kmeans = KMeans(n_clusters = k,n_init=10).fit(df_k)
        labels = kmeans.labels_
        sil.append(silhouette_score(df_k, labels, metric = 'euclidean'))
    optimal_k = kmeans_params_sil[np.argmax(sil)]
    print("Optimal number of clusters (K):", optimal_k)
    # perform Elbow Method analysis
    plt.figure(figsize=(16, 8))
    plt.plot(kmeans_params, inertia_scores, "bx-")
    plt.xlabel("k")
    plt.ylabel("Inertia scores")
    plt.title("The Elbow Method showing the optimal k")
    plt.show()
    
    # Plot Silhouette scores
    plt.figure(figsize=(16, 8))
    plt.plot(kmeans_params_sil, sil, "bx-")
    plt.xlabel("k")
    plt.ylabel("Silhouette scores")
    plt.title("Silhouette scores for different values of k")
    plt.show()
    model.fit(df)
    df["predicted"] = model.predict(df)
    # Compare actual cluster versus KMeans predicted cluster
    fig, ax = plt.subplots(1, 2, figsize=(16,8))
    ax[0].scatter(df.iloc[:, 0], df.iloc[:, 1], c=df["Class"], cmap=plt.cm.Set1)
    ax[1].scatter(df.iloc[:, 0], df.iloc[:, 1], c=df["predicted"], cmap=plt.cm.Set1)
    ax[0].set_title("Actual Clusters", fontsize=18)
    ax[1].set_title(f"KMeans Clusters (k={optimal_k})", fontsize=18)
    plt.show()
    
    return optimal_k
# print('K-optimal for original new dataset')
# k_optimal_df_model = find_optimal_k(df)
# print('K-optimal for Standardization new dataset')
# k_optimal_df_model = find_optimal_k(df_scaled)
# print('K-optimal for PCA new dataset')
# k_optimal_df_model = find_optimal_k(pca_df)
# print('K-optimal for PCA the Standardization new dataset')
# k_optimal_df_model = find_optimal_k(pca_df_std)


def evaluate_kmeans_clustering(df, n_clusters=None, max_iter=100, random_state=0):
    df = df.iloc[:, 1:]
    true_labels = df.iloc[:, 0].values.astype(int)
    features = df.iloc[:, 1:].values
    
    # Build K-Means Clustering model
    model = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)
    clusters = model.fit_predict(features)

    # Map cluster labels to class labels
    mapped_labels = []
    for cluster_id in range(n_clusters):
        cluster_indices = (clusters == cluster_id)
        true_labels_in_cluster = true_labels[cluster_indices]
        # Count the frequency of class labels in the cluster
        class_counts = np.bincount(true_labels_in_cluster)
        # Choose the most common class label in the cluster
        mapped_label = np.argmax(class_counts)
        mapped_labels.append(mapped_label)
    # Map cluster labels back to corresponding class labels
    mapped_labels = [mapped_labels[cluster_id] for cluster_id in clusters]

    # Compare actual class labels, raw K-Means labels, and mapped labels
    print("True class labels:\n", true_labels)
    print("K-Means cluster labels (raw):\n", model.labels_)
    print("K-Means cluster labels (mapped to class):\n", mapped_labels)

    # Calculate and print the accuracy of K-Means Clustering
    accuracy = accuracy_score(true_labels, mapped_labels)
    print("Overall K-Means accuracy: %.2f%%" % (accuracy * 100))

    # Visualize true class labels vs. K-Means clusters
    visualize_clusters(features, true_labels, clusters, model.cluster_centers_)
    # # visualise confusion matrix
    # mat = confusion_matrix(true_labels, mapped_labels)
    # sns.heatmap(mat.T, square=True, annot=True, fmt="d", cbar=False,
    #         xticklabels='digits.target_names',
    #         yticklabels=true_labels)
    # plt.xlabel("true label")
    # plt.ylabel("predicted label")
    # plt.show()

    return accuracy, mapped_labels

def visualize_clusters(features, true_labels, kmeans_labels, centroids):
    plt.figure(figsize=(12, 6))
    
    # Plot true class labels
    plt.subplot(1, 2, 1)
    plt.scatter(features[:, 0], features[:, 1], c=true_labels, cmap='viridis', edgecolor='k', s=50)
    plt.title('True Class Labels')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Plot K-Means clusters
    plt.subplot(1, 2, 2)
    plt.scatter(features[:, 0], features[:, 1], c=kmeans_labels, cmap='viridis', edgecolor='k', s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='o', s=200, alpha=0.8, label='Centroids')
    plt.title('K-Means Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.tight_layout()
    plt.show()


print('original')
df_cluster = evaluate_kmeans_clustering(df, 2)
print('std')
df_cluster_scaled = evaluate_kmeans_clustering(df_scaled, 2)
print('pca')
pca_df_cluster_scaled = evaluate_kmeans_clustering(pca_df, 2)
print('pca+std')
pca_std_df_cluster_scaled = evaluate_kmeans_clustering(pca_df_std, 2)