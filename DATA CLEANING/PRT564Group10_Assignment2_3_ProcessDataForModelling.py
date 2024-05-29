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

# Function to extract countries from affiliations
def extract_countries(affiliations):
    country_pattern = r',\s*([^,\d]+)$|,\s*([^,\d;]+);'
    # country_pattern = r',\s*([^,]+)$|,\s*([^,;]+);'
    
    # Find all matches of country pattern in affiliations
    matches = re.findall(country_pattern, affiliations)
    
    # Initialize an empty set to store unique countries
    countries = set()

    # Loop through the matches and add the non-empty country to the set
    for match in matches:
        # country = match[1].strip()  # Remove leading and trailing whitespace
        if match[0].strip():
            countries.add(match[0].strip())
        else:
            countries.add(match[1].strip())

    # Concatenate the unique countries using semicolons
    countries_concatenated = '; '.join(countries)
    
    return countries_concatenated

# def extract_countries(affiliations):
#     country_pattern = r',\s*([^,]+)$|,\s*([^,;]+);'
    
#     # Find all matches of country pattern in affiliations
#     matches = re.findall(country_pattern, affiliations)
    
#     # Initialize an empty set to store unique countries
#     countries = set()

#     # Loop through the matches and add the non-empty country to the set
#     for match in matches:
#         country = match[0].strip()  # Remove leading and trailing whitespace
#         if country:
#             countries.add(country)

#     # Concatenate the unique countries using semicolons
#     countries_concatenated = ';'.join(countries)
    
#     return countries_concatenated

# Read the CSV file consists of retracted papers
raw_df = pd.read_csv('download\data_2018_v1.0.csv', encoding='ISO-8859-1')
raw_df.columns = ['Author','Title','Year','Journal','Cited by', 'DOI', 'Institution','Publisher','Document Type','Publication Stage','Open Access','Record ID']
raw_df = raw_df[['Record ID', 'Author','Title','Year','Journal','Cited by', 'DOI', 'Institution','Publisher','Document Type','Publication Stage','Open Access']]
raw_df.drop_duplicates()
# Drop n/a value for institution
raw_df = raw_df.dropna(subset=['Institution'])

# Show total record for download data
count_record = raw_df['Record ID'].nunique()
# print("\nTOTAL RECORDS:", count_record)

# Show the total retraction record in download dataset
retracted_count = (raw_df['Document Type'] == 'Retracted').sum()
# print("\nRetracted Papers:", retracted_count)

# Show the total public paper record in download dataset
nonretracted_count = (raw_df['Document Type'] == 'Article').sum()
# print("\nNon-Retracted Papers:", nonretracted_count)
# Create DataFrame
data = {
    'TOTAL RECORD': [count_record],
    'No. of RETRACTED PAPERS': [retracted_count],
    'No. of NON-RETRACTED PAPERS': [nonretracted_count]
}
count_df = pd.DataFrame(data)

# Print DataFrame
print(count_df)

# Extract data for single country
raw_df['Country'] = raw_df['Institution'].apply(extract_countries) 
raw_df = raw_df.dropna(subset=['Country']) 

raw_df.to_csv('download/result_country.csv', index=False)
