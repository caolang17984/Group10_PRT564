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

# Create a function to count Record ID without duplicates
def count_unique_records(df, column_name):
    # Group by the specified column and count the number of unique 'Record ID'
    unique_counts = df.groupby(column_name)['Record ID'].nunique().reset_index(name='unique_count')
    return unique_counts

# Load the retraction data into a DataFrame
retractions_df = pd.read_csv('stagingdata_visualization.csv')

# Count the number of retracted papers for each year
retracted_papers_by_year = count_unique_records(retractions_df, 'Retraction_Year')

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.plot(retracted_papers_by_year['Retraction_Year'], retracted_papers_by_year['unique_count'], marker='o', color='blue')
plt.xlabel('Year')
plt.ylabel('Number of Retracted Papers')
plt.title('Number of Retracted Papers by Year')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Print the result
print(retracted_papers_by_year)
