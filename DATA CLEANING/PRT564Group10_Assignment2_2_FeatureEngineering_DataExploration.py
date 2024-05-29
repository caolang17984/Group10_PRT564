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

# Function to get title lenght
def get_title_length(title):
    return len(title.split())

#--------------------------------------------------------------------------------
# Read data from file
#--------------------------------------------------------------------------------
retractions_df = pd.read_csv('cleandata.csv')
# print(raw_df.columns)
raw_df = retractions_df[['Record ID', 'Title', 'Subject', 'Institution', 'Journal', 'Publisher',
       'Country', 'Author', 'ArticleType', 'RetractionDate', 'RetractionDOI', 'OriginalPaperDate','RetractionNature',
       'Reason', 'Paywalled', 'CitationCount']].copy()

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

#--------------------------------------------------------------------------------
# Extract number of Author
#--------------------------------------------------------------------------------
# Extract data for Author
# Count the number of authors in each row
raw_df['Author_Count'] = raw_df['Author'].str.count(';') + 1

#--------------------------------------------------------------------------------
# Feature Engineering: No. of Reason
#--------------------------------------------------------------------------------
# Extract data for Reason
raw_df['Reason_Count'] = raw_df['Reason'].str.count(';') + 1

# # Drop Retraction & Publication Dates
# raw_df.drop(columns=['RetractionDate', 'PublicDate'], inplace=True)

#--------------------------------------------------------------------------------
# Extract multiple value for country
#--------------------------------------------------------------------------------
# # Load the retraction data into a DataFrame
author_df = expand_rows(raw_df, 'Country')



# author_df = expand_rows(country_df, 'Author')
# author_df.to_csv("result/author_df.csv", index=False)


#--------------------------------------------------------------------------------
# Extract multiple value for Reason
#--------------------------------------------------------------------------------
# Extract data for Reason
reason_df = expand_rows(author_df, 'Reason')
reason_df.to_csv("result/reason_df.csv", index=False)

#--------------------------------------------------------------------------------
# Extract multiple value for Subject
#--------------------------------------------------------------------------------
# Extract data for Subject with full text
subject_df = expand_rows(reason_df, 'Subject')
subject_df.to_csv("result/subject_df.csv", index=False)

# Extract the short term and full text into separate columns
subject_df['Subject_Short_Term'] = subject_df['Subject_rep'].str.extract(r'\((.*?)\)')
subject_df['Subject_Full_Text'] = subject_df['Subject_rep'].str.extract(r'\(.*?\)\s(.*)')
subject_df.drop(columns=['Subject_rep'], inplace=True)

# # Extract data for Subject with short term
# subject_df = expand_rows_short_subject(long_subject_df, 'Subject')
subject_df.to_csv("result/subject_df.csv", index=False)

#--------------------------------------------------------------------------------
# Extract multiple value for Article Type
#--------------------------------------------------------------------------------
articletype_df = expand_rows(subject_df, 'ArticleType')
articletype_df.to_csv("result/articletype_df.csv", index=False)
articletype_df.to_csv("result/stagingdata_visualization.csv", index=False)

