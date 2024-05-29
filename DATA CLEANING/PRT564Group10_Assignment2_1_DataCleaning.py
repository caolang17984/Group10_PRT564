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
# Function to replace all unknow, unavailable of empty values
# and values have single charaters by N/A value
def remove_unknown_value(org_df, cols_name):
    org_clean_df = org_df.copy()
    for col_name in cols_name:
        # Replace "Unknown" and single-letter country names with NaN
        org_clean_df[col_name] = org_clean_df[col_name].replace(['Unknown', 'Unknown;', 'unavailable', 'unavailable;', r'\b\w\b'], pd.NA, regex=True)
           
    return org_clean_df

# Function to extract countries from affiliations
def extract_countries(affiliations):
    country_pattern = r',\s*([^,]+)$|,\s*([^,;]+);'
    
    # Find all matches of country pattern in affiliations
    matches = re.findall(country_pattern, affiliations)
    
    # Initialize an empty list to store countries
    countries = []

    # Loop through the matches and concatenate the non-empty country
    for match in matches:
        country = match[0] or match[1]  # Use the non-empty group
        if country:
            countries.append(country)

    # Concatenate the countries using semicolons
    countries_concatenated = ';'.join(countries)
    
    return countries_concatenated

# Function to merge missing data
def fill_missing_data(or_df, fm_df, author = False):  
    daff_df = fm_df.copy() 
    retraction_df = or_df.copy()
    # Apply the extract_countries function to create the "Country" column
    daff_df['Country'] = daff_df['Institution'].apply(extract_countries)   
         
    # Left join retraction_df with daff_df based on 'DOI' and 'OriginalPaperDOI'
    merged_df = pd.merge(retraction_df, daff_df, left_on='OriginalPaperDOI', right_on='DOI', how='left', suffixes=('', '_daff'))

    # Fill missing values in 'Institution' and 'Country' columns with values from daff_df
    merged_df['Institution'] = merged_df.apply(lambda row: row['Institution_daff'] if pd.isna(row['Institution']) else row['Institution'], axis=1)
    merged_df['Country'] = merged_df.apply(lambda row: row['Country_daff'] if pd.isna(row['Country']) else row['Country'], axis=1)
    
    if author:
        merged_df['Author'] = merged_df.apply(lambda row: row['Author_daff'] if pd.isna(row['Author']) else row['Author'], axis=1)
        # Drop the additional 'DOI' and 'Country_daff' columns if needed
        merged_df.drop(['DOI', 'Country_daff', 'Institution_daff', 'Author_daff'], axis=1, inplace=True) 

    else:
        # Drop the additional 'DOI' and 'Country_daff' columns if needed
        merged_df.drop(['DOI', 'Country_daff', 'Institution_daff'], axis=1, inplace=True) 
        # Remove duplicate rows
    merged_df = merged_df.drop_duplicates(subset=['Record ID'])
    
    return merged_df

# Function to count missing value
def count_missing_value(retraction_df):
    # Count missing values in 'Institution', 'Country', 'Author', 'Paywalled' columns
    missing_institution = retraction_df['Institution'].isna().sum()
    missing_country = retraction_df['Country'].isna().sum()
    missing_author = retraction_df['Author'].isna().sum()
    missing_paywalled = retraction_df['Paywalled'].isna().sum()
    # print("Number of missing values in 'Institution' column:", missing_institution)
    # print("Number of missing values in 'Country' column:", missing_country)
    # print("Number of missing values in 'Author' column:", missing_author)
    # print("Number of missing values in 'Paywalled' column:", missing_paywalled)
    # Create DataFrame for missing value counts
    missing_value_table = pd.DataFrame({
        'Column Name': ['Institution', 'Country', 'Author', 'Paywalled'],
        'No. of Missing Values': [missing_institution, missing_country, missing_author, missing_paywalled]
    })
    
    # Print the DataFrame as a table
    print(missing_value_table)
   
#--------------------------------------------------------------------------------
# Staging 0: Extract raw data
# - Read data from csv file
# - Remove unknown value
#--------------------------------------------------------------------------------
# Read the CSV file consists of retracted papers
raw_df = pd.read_csv('csv/retractions.csv', encoding='utf-8')
retraction_df = remove_unknown_value(raw_df, ['Institution', 'Country', 'Author', 'Paywalled'])

# Summary missing value before cleansing data
print("BEFORE CLEANING")
count_missing_value(retraction_df)
# Count the total report from csv file before cleaning
count_record = retraction_df['Record ID'].nunique()
print("\nTOTAL RECORDS:", count_record)


#--------------------------------------------------------------------------------
# Staging 1: Based on Missing Country, get Institution data based on Affiliation
#--------------------------------------------------------------------------------
# Read the CSV consists of DOI and Institution information which is download from scopus
daff_df = pd.read_csv('csv/doi_affiliation.csv', encoding='ISO-8859-1')
# Define the column names for daff_df dataframe
daff_df_columns = ['DOI', 'Institution']
daff_df.columns = daff_df_columns

staging1_df = fill_missing_data(retraction_df, daff_df)
# Summary missing value before cleansing data
print("\nAFTER FILLING MISSING VALUE FOR COUNTRY")
count_missing_value(staging1_df)


#--------------------------------------------------------------------------------
# Staging 2: Based on Missing Institution, get Institution data based on Affiliation
#--------------------------------------------------------------------------------
# Read the CSV file
daff_df = pd.read_csv('csv/doi_affiliation_Institution.csv', encoding='ISO-8859-1')
# Define the column names for daff_df dataframe
daff_df_columns = ['DOI', 'Institution']
daff_df.columns = daff_df_columns

staging2_df = fill_missing_data(staging1_df, daff_df)
# Summary missing value before cleansing data
print("\nAFTER FILLING MISSING VALUE FOR INSTITUTION")
count_missing_value(staging2_df)

#--------------------------------------------------------------------------------
# Staging 3: Based on Missing Author, get Institution data based on Affiliation
#--------------------------------------------------------------------------------
# Read the CSV file
daff_df = pd.read_csv('csv/doi_affiliation_author.csv', encoding='ISO-8859-1')
# Define the column names for daff_df dataframe
daff_df_columns = ['Author', 'DOI', 'Institution']
daff_df.columns = daff_df_columns

staging3_df = fill_missing_data(staging2_df, daff_df, True)
# Summary missing value before cleansing data
print("\nAFTER CLEANING")
count_missing_value(staging3_df)


#--------------------------------------------------------------------------------
# Stage 4: Drop record missing Institution, Country, Author and Paywalled
#--------------------------------------------------------------------------------
# Remove rows with missing values in 'Institution', 'Country', 'Author', or 'Paywalled' columns
staging3_df = staging3_df.dropna(subset=['Institution', 'Country', 'Author', 'Paywalled'])
staging3_df.drop_duplicates()
count_record = staging3_df['Record ID'].nunique()
print("\nTOTAL RECORDS:", count_record)

staging3_df.to_csv("result/cleandata.csv", index=False)


