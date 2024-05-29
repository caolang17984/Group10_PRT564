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
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd

def train_naive_bayes_original(df):
    # Separate response variable (y) from explanatory variables (X)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    
    # Training and test splits on original data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    #on original data
    clf = GaussianNB().fit(X_train, y_train)
    y_train_pred = clf.predict(X_test)
    print("\nPrediction accuracy for the test dataset:")
    print("{:.2%}".format(metrics.accuracy_score(y_test, y_train_pred)))

    return 

def train_naive_bayes_standardized(df):
    # Separate response variable (y) from explanatory variables (X)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    
    # Training and test splits on original data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # Apply standardization to explanatory variables
    std_scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_std = std_scaler.transform(X_train)
    X_test_std = std_scaler.transform(X_test)


    # Build Naive Bayes classifier on standardized data
    clf_std = GaussianNB().fit(X_train_std, y_train)
    y_train_pred_std = clf_std.predict(X_test_std)

    print("\nPrediction accuracy for the test dataset (with standardisation):")
    print("{:.2%}".format(metrics.accuracy_score(y_test, y_train_pred_std)))

    return 

def train_naive_bayes_pca(df, num_components_1,num_components_2):
    # Separate response variable (y) from explanatory variables (X)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    
    # Training and test splits on original data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # Apply standardization to explanatory variables
    std_scaler = preprocessing.StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)
    X_test_std = std_scaler.transform(X_test)

    # Initialize PCA
    pca1 = PCA(n_components=num_components_1)
  
    # fit a 2-component PCA on original training set
    pca1.fit(X_train)
    
    pca_std = PCA(n_components=num_components_2)
    pca_std.fit(X_train_std)
    # reduce dimensionality of training and test sets
    X_train_pca = pca1.transform(X_train)
    X_test_pca = pca1.transform(X_test)
    

    # Build Naive Bayes classifier on original data after PCA
    clf_pca = GaussianNB().fit(X_train_pca, y_train)
    y_train_pred_pca = clf_pca.predict(X_test_pca)

    print("\nPrediction accuracy the test dataset dataset:")
    print("{:.2%}".format(metrics.accuracy_score(y_test, y_train_pred_pca)))
    
    # reduce dimensionality of standardised training and test sets
    X_train_std_pca = pca_std.transform(X_train_std)
    X_test_pca_std = pca_std.transform(X_test_std)


    # Build Naive Bayes classifier on standardized data after PCA
    clf_std_pca = GaussianNB().fit(X_train_std_pca, y_train)
    y_train_pred_std_pca = clf_std_pca.predict(X_test_pca_std)

    print("\nPrediction accuracy for the training dataset (with standardisation):")
    print("{:.2%}".format(metrics.accuracy_score(y_test, y_train_pred_std_pca)))
    return 


# read a remote .csv file
df = pd.read_csv('df_model.csv')

df_naive = train_naive_bayes_original(df)

df_standardized = train_naive_bayes_standardized(df)

print('Applying PCA to test datasets')
df_pca = train_naive_bayes_pca(df,3,8)
