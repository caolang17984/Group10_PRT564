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
df = pd.read_csv('PCA_new_dataset_model_df.csv')

df.rename(columns={'Class': 'Class', 'PC1': 'Author', 'PC2': 'Journal', 'PC3': 'Title_length', 
                   'PC4': 'Mean_Citation','PC5': 'Rank','PC6': 'Country','PC7': 'Institution'}, inplace=True)
# print(df)
# Separate explanatory continuous variables

X = df.iloc[:,1::].values

y = df.iloc[:,0].values
#==========================================================================#
# Naive Bayes: Dicresitation + Test
#==========================================================================#

# Training and test splits on original data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)

estimators = [
    ("Random Forest Classifier", RandomForestClassifier(random_state=0))
]

param_sets = [
    # RFR hyperparameters
    {
        "n_estimators": [100, 250, 500, 750, 1000],
        "max_depth": [10, 50, 100]
    }]

# run GridSearchCV for each estimator and report its perfomance on the test set
for (est, params) in zip(estimators, param_sets):
    print("Finding the best hyperparameters for %s ..." % est[0])
    
    model = GridSearchCV(
        est[1],
        param_grid=params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
        # train
    model.fit(X_train, y_train)
    print("Best estimator parameters: ", model.best_params_)
    best_model = model.best_estimator_
    
    # evaluate on the test set
    test_score = best_model.score(X_test, y_test)
    print("Test set accuracy: ", test_score)



# specify classifiers to compare
classifiers = [
    ('Gaussian', GaussianNB()),
    ('SVM', SVC(kernel='linear')),
    ('Random Forest Classifier', RandomForestClassifier(n_estimators=250, random_state=0)),
    ('XGBoost', XGBClassifier()),
    ('Decision Tree', DecisionTreeClassifier())
]

# train each classifier in the list on the same training set
# evaluate its performance on the same test set
for name, clf in classifiers:
    
    # fit classifier 
    clf.fit(X_train, y_train)
    
    # make prediction
    y_hat = clf.predict(X_test)
 
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