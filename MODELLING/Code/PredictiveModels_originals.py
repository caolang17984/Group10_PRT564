import pandas as pd
import numpy as np

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


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
# X = D
# dis_df = apply_discretization(con_df)
# och_df = apply_one_hot_encoding(cat_df)
# # X = pd.concat([dis_df, och_df], axis=1)
X = df.iloc[:,1::].values
y = df.iloc[:,0].values
#==========================================================================#
# Naive Bayes: Dicresitation + Test
#==========================================================================#

# Training and test splits on original data
X_train_disc, X_test_disc, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)

# specify classifiers to compare
classifiers = [
    ('BernoulliNB', BernoulliNB()),
    ('SVM', SVC(kernel='linear')),
    ('Random Forest Classifier', RandomForestClassifier(n_estimators=1000, random_state=0)),
    ('XGBoost', XGBClassifier()),
    ('Decision Tree', DecisionTreeClassifier())
]


# train each classifier in the list on the same training set
# evaluate its performance on the same test set
for name, clf in classifiers:
    
    # fit classifier 
    clf.fit(X_train_disc, y_train)
    
    # make prediction
    y_hat = clf.predict(X_test_disc)
 
    # compute precision, recall, and F1-score
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_hat, average='binary')
    
    # compute accuracy('XGBoost', XGBClassifier())
    acc_score = accuracy_score(y_test, y_hat) * 100
        
    # print result
    print("\nClassifier: ", name)
    print("Accuracy: %.3f%%" % acc_score)
    print("Precision: %.3f%%" % (precision * 100))
    print("Recall: %.3f%%" % (recall * 100))
    print("F1 Score: %.3f%%" % (f1_score * 100))
    
    