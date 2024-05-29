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
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def group_reasons(reason):
    # Define groups based on commons starting word
    groups = {
        'Falsification/Fabrication': ['Falsification/Fabrication of Image', 'Falsification/Fabrication of Data', 'Falsification/Fabrication of Results'],
        'Original Data not Provided': ['Original Data not Provided'],
        'Taken from Dissertation/Thesis': ['Taken from Dissertation/Thesis'],
        'Concerns/Issues': ['Concerns/Issues about Referencing/Attributions','Concerns/Issues About Results','Concerns/Issues About Data','Concerns/Issues About Image'],
        'Duplication': ['Duplication of Image', 'Duplication of Text', 'Duplication of Data', 'Duplication of Article'],
        'Error': ['Error in Methods', 'Error in Analyses', 'Error in Data', 'Error in Materials (General)', 'Error in Results and/or Conclusions', 'Error in Image'],
        'Bias Issues or Lack of Balance': ['Bias Issues or Lack of Balance'],
        'Results Not Reproducible': ['Results Not Reproducible'],
        'Plagiarism': ['Plagiarism of Text', 'Plagiarism of Data', 'Plagiarism of Image', 'Plagiarism of Article'],
        'Unreliable': ['Unreliable Data', 'Unreliable Image', 'Unreliable Results'],
        'Manipulation': ['Manipulation of Results', 'Manipulation of Images'],
        'Randomly Generated Content': ['Randomly Generated Content'],
        'Euphemisms': ['Euphemisms for Plagiarism', 'Euphemisms for Duplication']
    }
    
    for group, reasons in groups.items():
        if reason in reasons:
            return group
    return reason

# Create a function to count Record ID without duplicates
def count_unique_records(df, column_name):
    # Group by the specified column and count the number of unique 'Record ID'
    unique_counts = df.groupby(column_name)['Record ID'].nunique().reset_index(name='unique_count')
    return unique_counts

# Read data from CSV files
reason_category_df = pd.read_csv('Reason_Category.csv')
retractions_df = pd.read_csv('stagingdata_visualization.csv')

# Merge the data
retractions_merged_df = pd.merge(retractions_df, reason_category_df, left_on='Reason_rep', right_on='Watch Retraction - Reason', how='left')

# Convert the Retraction_Year column to datetime format
retractions_merged_df['Retraction_Year'] = pd.to_datetime(retractions_merged_df['Retraction_Year'], format='%Y')

# Filter rows related to content and after the year 2000
content_retractions_df = retractions_merged_df[(retractions_merged_df['Reason Categories'] == 'Content') & (retractions_merged_df['Retraction_Year'].dt.year >= 2000)]

# Apply the grouping function to the reasons
content_retractions_df['Grouped Reason'] = content_retractions_df['Watch Retraction - Reason'].apply(group_reasons)

# Count the number of unique Record ID for each grouped reason
unique_counts = count_unique_records(content_retractions_df, 'Grouped Reason')

# Group data by year and grouped reason, then calculate the total number of retractions for each group
grouped_data = content_retractions_df.groupby([content_retractions_df['Retraction_Year'].dt.year, 'Grouped Reason'])['Record ID'].nunique().unstack().fillna(0)

# Plot a stacked bar chart
plt.figure(figsize=(14, 10)) 
sns.set_palette("tab20")  # Set palette to use distinct colors
grouped_data.plot(kind='bar', stacked=True)
plt.title('Number of Retracted Papers by Content-Related Reasons since 2000')
plt.xlabel('Year')
plt.ylabel('Number of Retractions')
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.legend(title='Content Reasons', bbox_to_anchor=(1, 1), loc='upper left', fontsize='small')  # Move legend to the side and make it smaller
plt.tight_layout()
plt.show()

# Print the result
print(grouped_data.to_string())



########################################################################################################

# Filter rows related to "Unreliable" group
unreliable_retractions_df = content_retractions_df[content_retractions_df['Grouped Reason'] == 'Unreliable']

# Group data by specific reason within "Unreliable" group
grouped_unreliable_data = unreliable_retractions_df.groupby('Watch Retraction - Reason')['Record ID'].nunique().reset_index(name='count')

# Plot a pie chart for "Unreliable" group breakdown by specific reasons
plt.figure(figsize=(10, 8))
wedges, texts, autotexts = plt.pie(grouped_unreliable_data['count'], labels=grouped_unreliable_data['Watch Retraction - Reason'], autopct='%1.1f%%', startangle=140, pctdistance=0.85)

# Customise the pie chart
for text in texts:
    text.set_fontsize(10)
for autotext in autotexts:
    autotext.set_fontsize(10)

plt.title('Breakdown of "Unreliable" Group of Reasons in Content-Related Category')
plt.axis('equal') 
plt.tight_layout()
plt.show()

# Print the result
print(grouped_unreliable_data)