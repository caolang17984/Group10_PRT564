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

# Load the retraction data into a DataFrame
retractions_df = pd.read_csv('stagingdata_visualization.csv')

# Count the number of retracted papers by number of author
retracted_papers_by_author = count_unique_records(retractions_df, 'Author_Count')



# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(retracted_papers_by_author['Author_Count'], retracted_papers_by_author['unique_count'], color='blue')
plt.xlabel('Number of Author')
plt.ylabel('Number of Retracted Papers')
plt.title('Number of Retracted Papers by Number of Author')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Print the result
print(retracted_papers_by_author)

################ P I E C H A R T ################

# Define a function to categorise author counts
def categorise_author_counts(author_count):
    if author_count <= 5:
        return str(author_count) + ' authors'
    else:
        return 'More than 5 authors'

# Apply categorisation to create a new column
retracted_papers_by_author['Author_Group'] = retracted_papers_by_author['Author_Count'].apply(categorise_author_counts)

# Summarise the counts by the new categories
grouped_counts = retracted_papers_by_author.groupby('Author_Group')['unique_count'].sum().reset_index()

# Create a donut chart for the unique counts
plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(grouped_counts['unique_count'], labels=grouped_counts['Author_Group'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette("tab20", len(grouped_counts)), wedgeprops=dict(width=0.3))

# Draw a circle at the center to make it a donut chart
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)


plt.title('The Proportion of Number of Authors')
plt.axis('equal')
plt.show()
