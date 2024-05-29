import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score 
from sklearn.model_selection import train_test_split, GridSearchCV
# read a remote .csv file
def apply_discretization(df, n_bins=2):
    # Initialize the discretizer
    disc = KBinsDiscretizer(n_bins=n_bins)
    # Apply the discretizer to each column and store results
    discretized_data = pd.DataFrame()
   
    for column in df.columns:
        col_data = df[column].values
        col_data = np.reshape(col_data,(len(col_data),1))
        discretized_col = disc.fit_transform(col_data).toarray()
        # Convert the array to a DataFrame
        discretized_col_df = pd.DataFrame(discretized_col, columns=[f"{column}_{i}" for i in range(discretized_col.shape[1])])
        # Concatenate the discretized column to the discretized_data DataFrame
        discretized_data = pd.concat([discretized_data, discretized_col_df], axis=1)
    return discretized_data

def apply_one_hot_encoding(cat_df):
    enc = OneHotEncoder(sparse=False)
    # Initialize an empty DataFrame to store discretized data
    ohc_data = pd.DataFrame()
    for column in cat_df.columns:
        # Reshape the data to (-1, 1)
        data = cat_df[column].values.reshape(-1, 1)
        # Apply one-hot encoding
        transformed_data = enc.fit_transform(data)
        # Convert the encoded features into a DataFrame
        encoded_df = pd.DataFrame(transformed_data, 
                                  columns=[f"{column}_{int(i)}" for i in range(transformed_data.shape[1])])
        # Concatenate the new DataFrame to the original DataFrame
        ohc_data = pd.concat([ohc_data, encoded_df], axis=1)
        
    return ohc_data

# read a remote .csv file
df = pd.read_csv('df_new_dataset_model.csv')
# Rename the columns
df.rename(columns={'Class': 'Class', 'Title Length': 'Title_Length', 'Country_Count': 'Country_Count', 'Author_Count': 'Author_Count', 
                   'Institution_Count': 'Institution_Count','Journal_encodeder': 'Journal','Publisher_encodeder': 'Publisher','Rank_group': 'Rank','CIT_Avg.':'Mean_Citation'}, inplace=True)
# print(df)


# apply discretisation
disc = KBinsDiscretizer(n_bins=2)

# discretise title length
tit_len = df['Title_Length'].values
# print('title value before dicretisation', tit_len)
tit_len = np.reshape(tit_len, (len(tit_len), 1))
tit_len = disc.fit_transform(tit_len).toarray()
# print('title value after dicretisation', tit_len)

# discretise cited by
cite = df['Mean_Citation'].values
cite = np.reshape(cite, (len(cite), 1))
cite = disc.fit_transform(cite).toarray()

# discretise Country_Count
country_c = df['Country_Count'].values
country_c = np.reshape(country_c, (len(country_c), 1))
country_c = disc.fit_transform(country_c).toarray()

# discretise Author_Count
auth_c = df['Author_Count'].values
auth_c = np.reshape(auth_c, (len(auth_c), 1))
auth_c = disc.fit_transform(auth_c).toarray()

# discretise Institution_Count
ins_c = df['Institution_Count'].values
ins_c = np.reshape(ins_c, (len(ins_c), 1))
ins_c = disc.fit_transform(ins_c).toarray()


# apply one hot encoding
enc = OneHotEncoder()

# one-hot encode Rank_group
rg = df['Rank'].values
# print(rg)
rg = np.reshape(rg, (len(rg), 1))
rg = enc.fit_transform(rg).toarray()
# print(rg)

# one-hot encode Journal_encodeder
je = df['Journal'].values
je = np.reshape(je, (len(je), 1))
je = enc.fit_transform(je).toarray()
# print('PA test: ',je)

# one-hot encode Publisher_encodeder
pe = df['Publisher'].values
pe = np.reshape(pe, (len(pe), 1))
pe = enc.fit_transform(pe).toarray()

# split explanatory variables (X) from the response variable (y)
D = np.concatenate([cite,tit_len, country_c, auth_c, ins_c, rg, je, pe], axis=1)
X = D

y = df.iloc[:,0].values
#==========================================================================#
# Naive Bayes: Dicresitation + Test
#==========================================================================#

# Training and test splits on original data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)



param_sets = [100, 250, 500, 750, 1000]

results = {
    'Estimators': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 score': []
}

for n_estimators in param_sets:
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    
    # Train model
    clf.fit(X_train, y_train)
    
    # Prdict Y
    y_hat = clf.predict(X_test)
 
   # Calculate Precision, recall, F1- score
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_hat, average='binary')
    
    # Calculate accuracy
    acc_score = accuracy_score(y_test, y_hat)*100
    
    # Store the result
    results['Estimators'].append(f'{n_estimators}')
    results['Accuracy'].append(acc_score)
    results['Precision'].append(precision*100)
    results['Recall'].append(recall*100)
    results['F1 score'].append(f1*100)
    
    # Print the result
    print(f"\nClassifier: Random Forest Classifier with {n_estimators} estimators")
    print(f"Accuracy: %.3f%%" % acc_score)
    print(f"Precision: %.3f%%" % (precision * 100))
    print(f"Recall: %.3f%%" % (recall * 100))
    print(f"F1 Score: %.3f%%" % (f1 * 100))
    
df_results = pd.DataFrame(results)
print("\nPerformance Table:")
print(df_results)