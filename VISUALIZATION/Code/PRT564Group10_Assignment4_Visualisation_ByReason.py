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

# Create a function to count Record ID without duplicates
def count_unique_records(df, column_name):
    # Group by the specified column and count the number of unique 'Record ID'
    unique_counts = df.groupby(column_name)['Record ID'].nunique().reset_index(name='unique_count')
    return unique_counts


# Read data from CSV files
reason_category_df = pd.read_csv('Reason_Category.csv')
retractions_df = pd.read_csv('stagingdata_visualization.csv')

# Classify reasons into groups based on 'Reason_Category'
# Merge retractions_df with reason_category_df to have the 'Reason Categories' column
retractions_merged_df = pd.merge(retractions_df, reason_category_df, left_on='Reason_rep', right_on='Watch Retraction - Reason', how='left')

# Convert the RetractionDate column to datetime format
retractions_merged_df['Retraction_Year'] = pd.to_datetime(retractions_merged_df['Retraction_Year'], format='%Y')

# Count the number of unique Record ID for each reason category
unique_counts = count_unique_records(retractions_merged_df, 'Reason Categories') 

# Print the unique counts
print(unique_counts)

# Create a donut chart for the unique counts
plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(unique_counts['unique_count'], labels=unique_counts['Reason Categories'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette("tab20", len(unique_counts)), wedgeprops=dict(width=0.3))

# Draw a circle at the center to make it a donut chart
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Proportions of Reason Categories')
plt.axis('equal')
plt.show()

# Create a chart
plt.figure(figsize=(10, 6))
sns.set_palette("tab20")  

grouped_data = retractions_merged_df.groupby([retractions_merged_df['Retraction_Year'].dt.year // 10 * 10, 'Reason Categories'])['Record ID'].nunique().unstack()

# Plot trend line for each Reason Category
for column in grouped_data.columns:
    plt.plot(grouped_data.index, grouped_data[column], marker='o', label=column)

plt.title('Number of Retracted Papers by Reason Count')
plt.xlabel('Year')
plt.ylabel('Number of Retractions')
plt.legend()
plt.grid(axis='y')
plt.xticks(grouped_data.index, rotation=45)
plt.tight_layout()
plt.show()

# Print the result
print(grouped_data.to_string())
