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
retractions_df = pd.read_csv('cleandata.csv')
# print(raw_df.columns)
raw_df = retractions_df[['Record ID', 'Title', 'Subject', 'Institution', 'Journal', 'Publisher',
       'Country', 'Author', 'ArticleType', 'RetractionDate','OriginalPaperDate',
       'Reason', 'Paywalled', 'CitationCount']].copy()

# Remove the last semicolon from specified columns
columns_to_clean = ['Reason', 'Subject', 'Institution']
for column_name in columns_to_clean:
    raw_df[column_name] = raw_df[column_name].str.rstrip(';')
    # Remove the plus sign from each reason
    if column_name == "Reason":
        # retractions_df[column_name] = retractions_df[column_name].str.replace('+', '')
        raw_df[column_name] = raw_df[column_name].str.replace('+', '', regex=False)

#--------------------------------------------------------------------------------
# Feature Engineering: Title Length
#--------------------------------------------------------------------------------
raw_df['Title Length'] = raw_df['Title'].apply(get_title_length)

# # Drop Title column
# raw_df.drop(columns=['Title'], inplace=True)
#--------------------------------------------------------------------------------
# Feature Engineering: Elapsed Time
#--------------------------------------------------------------------------------
# Convert 'RetractionDate' and 'OriginalPaperDate' columns to datetime
raw_df['RetractionDate'] = pd.to_datetime(raw_df['RetractionDate'], format='%d/%m/%Y')
raw_df['OriginalPaperDate'] = pd.to_datetime(raw_df['OriginalPaperDate'], format='%d/%m/%Y')

# Calculate the difference in days and add a new column with a meaningful name
raw_df['Elapsed_Day'] = (raw_df['RetractionDate'] - raw_df['OriginalPaperDate']).dt.days
# Calculate the difference in days and convert it to weeks
raw_df['Elapsed_Weeks'] = round(((raw_df['RetractionDate'] - raw_df['OriginalPaperDate']).dt.days) / 7,0)

# Extract the year into a new column
raw_df['Retraction_Year'] = raw_df['RetractionDate'].dt.year
raw_df['Public_Year'] = raw_df['OriginalPaperDate'].dt.year

# Save to csv file for reference
# raw_df.to_csv("result/temp/raw_df.csv", index=False)

#--------------------------------------------------------------------------------
# Feature Engineering: No. of contributed country
#--------------------------------------------------------------------------------
raw_df ['Country_Count'] = raw_df['Country'].str.count(';') + 1

#--------------------------------------------------------------------------------
# Feature Engineering: No. of Author
#--------------------------------------------------------------------------------
# Extract data for Author
raw_df['Author_Count'] = raw_df['Author'].str.count(';') + 1


#--------------------------------------------------------------------------------
# Feature Engineering: No. of Reason
#--------------------------------------------------------------------------------
# Extract data for Reason
raw_df['Reason_Count'] = raw_df['Reason'].str.count(';') + 1


#--------------------------------------------------------------------------------
# Feature Engineering: Each reason is an exploratory variable
#--------------------------------------------------------------------------------
reason_df, reason_list = separating_column(raw_df)

# Show the ist of Reasons
print("Reason columns:")
print(reason_list)

# Save to csv file for reference
reason_df.to_csv("result/reason_df.csv", index=False)

#--------------------------------------------------------------------------------
# Feature Engineering: Each subject is an exploratory variable
#--------------------------------------------------------------------------------
# Extract data for Subject with full text
subject_df, subject_list = separating_column_subject(reason_df, 'Subject')

# Show the ist of Reasons
print("Subject columns:")
print(subject_list)

# Save to csv file for reference
subject_df.to_csv("result/subject_df.csv", index=False)

#--------------------------------------------------------------------------------
# Feature Engineering: Ex article type is an exploratory variable
#--------------------------------------------------------------------------------
articletype_df, atype_list = separating_column(subject_df, 'ArticleType')

# Show the ist of Reasons
print("Artcile Type columns:")
print(atype_list)

# Save to csv file for reference
articletype_df.to_csv("result/articletype_df1.csv", index=False)
articletype_df.to_csv("result/stagingdata_model1.csv", index=False)

