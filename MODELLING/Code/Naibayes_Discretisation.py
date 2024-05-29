from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
import numpy as np


# read a remote .csv file
df = pd.read_csv('df_model.csv')
df = df[['Class','Cited by','Title Length','Country_Count','Author_Count','Institution_Count','Journal_encodeder','Publisher_encodeder','Rank_group']] 
# print(df)
con_df = df.iloc[:, 6:]
# print(con_df)

# apply discretisation
disc = KBinsDiscretizer(n_bins=8)


# discretise title length
tit_len = df['Title Length'].values
# print('title value before dicretisation', tit_len)
tit_len = np.reshape(tit_len, (len(tit_len), 1))
tit_len = disc.fit_transform(tit_len).toarray()
# print('title value after dicretisation', tit_len)

# discretise cited by
cite = df['Cited by'].values
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
rg = df['Rank_group'].values
# print(rg)
rg = np.reshape(rg, (len(rg), 1))
rg = enc.fit_transform(rg).toarray()
# print(rg)

# one-hot encode Journal_encodeder
je = df['Journal_encodeder'].values
je = np.reshape(je, (len(je), 1))
je = enc.fit_transform(je).toarray()
# print('PA test: ',je)

# one-hot encode Publisher_encodeder
pe = df['Publisher_encodeder'].values
pe = np.reshape(pe, (len(pe), 1))
pe = enc.fit_transform(pe).toarray()

# split explanatory variables (X) from the response variable (y)
X = np.concatenate([tit_len, country_c, auth_c, ins_c, rg, je, pe], axis=1)
y = df.iloc[:,0].values
#==========================================================================#
# Naive Bayes: Dicresitation + Test
#==========================================================================#

# Training and test splits on original data
X_train_disc, X_test_disc, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Build Naive Bayes classifier on standardized data
clf_disc = BernoulliNB().fit(X_train_disc, y_train)
y_train_pred_std = clf_disc.predict(X_train_disc)

print("\nNaive Bayes Model: Prediction accuracy for the training dataset (with discretisation):")
print("{:.2%}".format(metrics.accuracy_score(y_train, y_train_pred_std)))



#==========================================================================#
# Naive Bayes: Dicresitation + Training
#==========================================================================#

# Training and test splits on original data
X_train_disc, X_test_disc, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Build Naive Bayes classifier on standardized data
clf_disc = BernoulliNB().fit(X_train_disc, y_train)
y_test_pred_std = clf_disc.predict(X_test_disc)

print("\nNaive Bayes Model: Prediction accuracy for the test dataset (with discretisation):")
print("{:.2%}".format(metrics.accuracy_score(y_test, y_test_pred_std)))

