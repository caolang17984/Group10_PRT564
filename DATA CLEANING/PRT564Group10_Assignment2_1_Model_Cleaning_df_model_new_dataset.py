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
import datetime
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
def rank_and_group(df, column_name):
    # Define the group boundaries
    group_boundaries = np.arange(1000, 9001, 1000)

    # Create a new column for the groups
    df[column_name + '_group'] = np.digitize(df[column_name], bins=group_boundaries) + 1

    # Any value greater than 9000 or None is assigned to group 0
    df.loc[df[column_name] > 9000, column_name + '_group'] = 0
    df.loc[df[column_name].isna(), column_name + '_group'] = 0

    return df

def stratified_sampling(df, column_name, sample_size):
    # Create an empty DataFrame to store the sampled data
    sampled_df = pd.DataFrame(columns=df.columns)
    
    # Calculate the number of samples to be taken from each category
    category_counts = df[column_name].value_counts()
    category_sample_sizes = (category_counts / category_counts.sum() * sample_size).astype(int)
    
    # Iterate over the unique values of the stratifying column
    for category, size in category_sample_sizes.items():
        # Sample data for the current category
        category_samples = df[df[column_name] == category].sample(n=size, random_state=42)
        
        # Append the sampled data to the result DataFrame
        sampled_df = pd.concat([sampled_df, category_samples])
    
    return sampled_df
#--------------------------------------------------------------------------------
# Process data for non-retracted papers
#--------------------------------------------------------------------------------
non_retracted_df = pd.read_csv('download/download_data_model_v1.0.csv')

#print(df)
#--------------------------------------------------------------------------------
# Feature Engineering: No. of Institution
#--------------------------------------------------------------------------------
# Extract data for Institution
non_retracted_df['Institution_Count'] = non_retracted_df['Institution'].str.count(';') + 1
#--------------------------------------------------------------------------------
# Feature Engineering: ENCODER THE NEW COLUMN FROM JOURNAL
#--------------------------------------------------------------------------------
label_encoder = LabelEncoder()
non_retracted_df['Journal_encodeder'] = label_encoder.fit_transform(non_retracted_df['Journal'])
journal_counts = non_retracted_df['Journal'].value_counts()
top_10_journal = journal_counts.nlargest(10)
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Feature Engineering: ENCODER THE NEW COLUMN FROM PUBLISHER
#--------------------------------------------------------------------------------
label_encoder = LabelEncoder()
non_retracted_df['Publisher_encodeder'] = label_encoder.fit_transform(non_retracted_df['Publisher'])
pub_counts = non_retracted_df['Publisher'].value_counts()
top_10_publisher = pub_counts.nlargest(10)
# #--------------------------------------------------------------------------------
non_retracted_df = rank_and_group(non_retracted_df,'Rank')

# print(df)
non_retracted_df = non_retracted_df[non_retracted_df['Document Type']=='Article']
non_retracted_df = stratified_sampling(non_retracted_df,'Journal_encodeder', 320)
#--------------------------------------------------------------------------------
# Feature Engineering: Calculate average citation from the public paper to current year
#--------------------------------------------------------------------------------
# Get the current year
current_year = datetime.datetime.now().year
# Calculate the CIT avg
non_retracted_df['CIT_Avg.'] = non_retracted_df['Cited by']/(current_year - non_retracted_df['Public_Year'])



#--------------------------------------------------------------------------------
# Process data for retracted papers
#--------------------------------------------------------------------------------
retracted_df = pd.read_csv('download/retracted_model.csv')

#--------------------------------------------------------------------------------
# Feature Engineering: No. of Institution
#--------------------------------------------------------------------------------
# Extract data for Institution
retracted_df['Institution_Count'] = retracted_df['Institution'].str.count(';') + 1
#--------------------------------------------------------------------------------
# Feature Engineering: ENCODER THE NEW COLUMN FROM JOURNAL
#--------------------------------------------------------------------------------
label_encoder = LabelEncoder()
retracted_df['Journal_encodeder'] = label_encoder.fit_transform(retracted_df['Journal'])
journal_counts = retracted_df['Journal'].value_counts()
top_10_journal = journal_counts.nlargest(10)
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Feature Engineering: ENCODER THE NEW COLUMN FROM PUBLISHER
#--------------------------------------------------------------------------------
label_encoder = LabelEncoder()
retracted_df['Publisher_encodeder'] = label_encoder.fit_transform(retracted_df['Publisher'])
pub_counts = retracted_df['Publisher'].value_counts()
top_10_publisher = pub_counts.nlargest(10)
# #--------------------------------------------------------------------------------
retracted_df = rank_and_group(retracted_df,'Rank')

# print(df)
retracted_df = retracted_df[retracted_df['Document Type']=='Retracted']
retracted_df = stratified_sampling(retracted_df,'Journal_encodeder', 616)
#--------------------------------------------------------------------------------
# Feature Engineering: Calculate average citation from the public paper to current year
#--------------------------------------------------------------------------------
# Get the current year
current_year = datetime.datetime.now().year
# Calculate the CIT_Avg. column, considering the condition
mask = (retracted_df['Retraction_Year'] - retracted_df['Public_Year']) != 0
retracted_df.loc[mask, 'CIT_Avg.'] = retracted_df.loc[mask, 'Cited by'] / (retracted_df.loc[mask, 'Retraction_Year'] - retracted_df.loc[mask, 'Public_Year'])
retracted_df.loc[~mask, 'CIT_Avg.'] = 0

complete_data_df = pd.concat([non_retracted_df, retracted_df], ignore_index=True)

# Replace values in 'Document Type' column and explicitly specify data type
complete_data_df['Class'] = complete_data_df['Document Type'].replace({'Retracted': 1, 'Article': 0}).astype(int)
print(complete_data_df.columns)


columns_model = ['Class','Rank_group', 'Title Length', 'Country_Count',
       'Author_Count', 'Institution_Count', 'Journal_encodeder',
       'Publisher_encodeder', 'CIT_Avg.']

# ['Class','Cited by', 'Rank_group', 'Title Length', 'Country_Count',
#        'Author_Count', 'Institution_Count', 'Journal_encodeder',
#        'Publisher_encodeder']

complete_data_df =complete_data_df[columns_model]


complete_data_df.to_csv('result/df_model.csv', index= False)