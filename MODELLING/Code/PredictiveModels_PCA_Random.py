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