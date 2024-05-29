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
import re

#--------------------------------------------------------------------------------
# Define common function
#--------------------------------------------------------------------------------
# Function to extract subject code   
def extract_subject_codes(subject):
    # Use regular expression to find content within brackets
    codes = re.findall(r'\((.*?)\)', subject)
    return list(set(codes)) 

# Function to extract multiple values in the same cells
def expand_rows(retractions_df, column_name, delimiter=';'):
    # Initialize an empty list to store new rows
    new_rows = []

    # Iterate over each row in the DataFrame
    for idx, row in retractions_df.iterrows():
        new_row = row.copy()  # Create a copy of the original row
        
        # Iterate over each specified column
        
        col_rep = column_name + '_rep'
        new_row[col_rep] = new_row[column_name]
        
        # Remove the last delimiter from specified columns
        new_row[col_rep] = new_row[col_rep].rstrip(delimiter)
        new_row[col_rep] = new_row[col_rep].replace('+', '')
            
        # Split values in each specified column by delimiter and create new rows
        values = new_row[col_rep].split(delimiter)
        for value in values:
            if value.strip():
                # Update the value in the new row
                new_row[col_rep] = value.strip()                
                new_rows.append(new_row.copy())  # Append the new row to the list

    # Create a new DataFrame from the list of new rows
    expanded_df = pd.DataFrame(new_rows)
    expanded_df.drop_duplicates(inplace=True)
    expanded_df.dropna(subset=[column_name], inplace=True)
        
    return expanded_df 

# Function to extract multiple values in the same cells
def expand_rows_short_subject(retractions_df, column_name, delimiter=';'):
    # Initialize an empty list to store new rows
    new_rows = []

    # Iterate over each row in the DataFrame
    for idx, row in retractions_df.iterrows():
        new_row = row.copy()  # Create a copy of the original row
        
        # Iterate over each specified column
        
        col_rep = column_name + '_rep'
        new_row[col_rep] = new_row[column_name]
        
        # Remove the last delimiter from specified columns
        new_row[col_rep] = new_row[col_rep].rstrip(delimiter)
        new_row[col_rep] = new_row[col_rep].replace('+', '')
            
        # Split values in each specified column by delimiter and create new rows
        values = new_row[col_rep].split(delimiter)
        for value in values:
            if value.strip():
                # Update the value in the new row
                new_row[col_rep] = value.strip()
                
                codes = extract_subject_codes(value)
                for code in codes:
                    new_row[col_rep] = code
                    new_rows.append(new_row.copy())  # Append the new row to the list
                
    # Create a new DataFrame from the list of new rows
    expanded_df = pd.DataFrame(new_rows)
    expanded_df.drop_duplicates(inplace=True)
    expanded_df.dropna(subset=[column_name], inplace=True)
        
    return expanded_df 
   

# Function to separating column data
def separating_column(retractions_df, column_name='Reason', delimiter=';'):
    unique_values_set = set()
    for subset in retractions_df[column_name]:
        for value in subset.split(delimiter):
            if value.strip():  # Check if the stripped value is not empty
                unique_values_set.add("["+column_name+"] "+value.strip())
        
    unique_values_set.discard('')
    
    for value in unique_values_set:
        retractions_df = retractions_df.copy()
        retractions_df.loc[:, value] = retractions_df[column_name].apply(lambda x: 1 if value in x.split(delimiter) else 0)

    return retractions_df, unique_values_set

# Function to get title lenght
def get_title_length(title):
    return len(title.split())

# Function to separating column data for subject
def separating_column_subject(retractions_df, column_name='Subject', delimiter=';'):
    unique_values_set = set()
    
    # return retractions_df
    if column_name == 'Subject':
        # Use regular expression to extract values within brackets
        pattern = re.compile(r'\((.*?)\)')
        for subset in retractions_df[column_name]:
            matches = pattern.findall(subset)
            for value in matches:
                unique_values_set.add(value.strip())
    else:
        for subset in retractions_df[column_name]:
            for value in subset.split(delimiter):
                unique_values_set.add(value.strip())
    
    unique_values_set.discard('')
    
    for value in unique_values_set:
        retractions_df = retractions_df.copy()
        retractions_df.loc[:, value] = retractions_df[column_name].apply(lambda x: 1 if value in x.split(delimiter) else 0)
    
    return retractions_df, unique_values_set

#--------------------------------------------------------------------------------
# Read data from file
#--------------------------------------------------------------------------------
data_df = pd.read_csv('download/result_country.csv', encoding='utf-8')
journalRank_df = pd.read_csv('download/journal_rank_2023.csv', encoding='utf-8')
# Left join retraction_df with journalRank_df based on 'Journal'
alldata_df = pd.merge(data_df, journalRank_df, left_on='Journal', right_on='Journal', how='left', suffixes=('', '_rank'))
# print(alldata_df.columns)
# print(raw_df.columns)
raw_df = alldata_df[['Record ID', 'Author','Title','Year','Journal','Cited by', 'DOI', 'Institution','Publisher','Document Type', 'Country', 'Rank']].copy()
raw_df['Retraction_Year'] = ''
raw_df = raw_df[['Record ID', 'Author','Title','Year', 'Retraction_Year', 'Journal','Cited by', 'DOI', 'Institution','Publisher','Document Type', 'Country', 'Rank']]
raw_df.columns = ['Record ID', 'Author','Title','Public_Year', 'Retraction_Year', 'Journal','Cited by', 'DOI', 'Institution','Publisher','Document Type', 'Country', 'Rank']
print(raw_df.columns)

#--------------------------------------------------------------------------------
# Feature Engineering: Title Length
#--------------------------------------------------------------------------------
raw_df['Title Length'] = raw_df['Title'].apply(get_title_length)

#--------------------------------------------------------------------------------
# Feature Engineering: No. of contributed country
#--------------------------------------------------------------------------------
raw_df['Country_Count'] = raw_df['Country'].str.count(';').fillna(0) + 1

#--------------------------------------------------------------------------------
# Feature Engineering: No. of Author
#--------------------------------------------------------------------------------
# Extract data for Author
raw_df['Author_Count'] = raw_df['Author'].str.count(';') + 1

#--------------------------------------------------------------------------------
# Feature Engineering: No. of Institution
#--------------------------------------------------------------------------------
# Extract data for Institution
raw_df['Inst_Count'] = raw_df['Institution'].str.count(';') + 1
print(raw_df.columns)
raw_df.to_csv("download/download_data_model_v1.0.csv", index=False)


# # Get Retraction year
# retracted_data_df = pd.read_csv('result/stagingdata_visualization.csv', encoding='utf-8')
# retracted_data_df = retracted_data_df[['RetractionDOI', 'Retraction_Year']]
# retracted_data_df.columns=['DOI', 'Retraction_Year']
# print(retracted_data_df)

# # Performing inner join
# raw_df = pd.merge(raw_df, retracted_data_df, on='DOI', how='inner')
# raw_df.to_csv("download/download_data_model_v1.0.csv", index=False)
# # Filter the records based on the condition
# condition = (raw_df['Document Type'] == 'Retracted') & (raw_df['Retraction_Year'].isnull())

# # Drop the filtered records
# raw_df = raw_df[~condition]

# raw_df.to_csv("download/download_data_model_v1.1.csv", index=False)