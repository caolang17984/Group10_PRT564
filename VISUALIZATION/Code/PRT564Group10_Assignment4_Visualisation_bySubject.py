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
import matplotlib.pyplot as plt
import seaborn as sns


# Create a function to count Record ID without duplicates
def count_unique_records(df, column_name):
    # Group by the specified column and count the number of unique 'Record ID'
    unique_counts = df.groupby(column_name)['Record ID'].nunique().reset_index(name='unique_count')
    return unique_counts

# Read data from CSV files
retractions_df = pd.read_csv('stagingdata_visualization.csv')

# Convert the RetractionDate column to datetime format
retractions_df['Retraction_Year'] = pd.to_datetime(retractions_df['Retraction_Year'], format='%Y')

# Count the number of unique Record ID for each reason category
unique_counts = count_unique_records(retractions_df, 'Subject_Short_Term') 

# Print the unique counts
print(unique_counts)

# Create a donut chart for the unique counts
plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(unique_counts['unique_count'], labels=unique_counts['Subject_Short_Term'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette("tab20", len(unique_counts)), wedgeprops=dict(width=0.3))

# Draw a circle at the center to make it a donut chart
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('The Proportion of Subjects')
plt.axis('equal')
plt.show()

# Create a chart
plt.figure(figsize=(10, 6))
sns.set_palette("tab30")  


grouped_data = retractions_df.groupby([retractions_df['Retraction_Year'].dt.year // 10 * 10, 'Subject_Short_Term'])['Record ID'].nunique().unstack()

# Plot trend line for each Reason Category
for column in grouped_data.columns:
    plt.plot(grouped_data.index, grouped_data[column], marker='o', label=column)

plt.title('Number of Retracted Papers by Subject')
plt.xlabel('Year')
plt.ylabel('Number of Retractions')
plt.legend()
plt.grid(axis='y')
plt.xticks(grouped_data.index, rotation=45)
plt.tight_layout()
plt.show()

# Print the result
print(grouped_data.to_string())

#################### p i e c h a r t  ####################

# Group by 'Subject_Short_Term' and 'Subject_Full_Text' and count the number of unique 'Record ID'
grouped_counts = retractions_df.groupby(['Subject_Short_Term', 'Subject_Full_Text'])['Record ID'].nunique().reset_index(name='count')

# Calculate proportions for each subject
subject_sizes = grouped_counts.groupby('Subject_Short_Term')['count'].sum()

# Determine the layout of the subplots grid based on the number of unique subjects
num_subjects = len(subject_sizes)
cols = 2  # Set the number of columns for the grid
rows = (num_subjects + cols - 1) // cols  # Calculate the number of rows needed

# Create subplots
fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 8))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop through all subjects in the dataset and create pie charts
for i, (subject, size) in enumerate(subject_sizes.items()):
    ax = axes[i]
    subject_data = grouped_counts[grouped_counts['Subject_Short_Term'] == subject]
    wedges, texts = ax.pie(subject_data['count'], labels=None, pctdistance=0.85, labeldistance=1.2, startangle=140, wedgeprops=dict(width=0.3))
    ax.set_title(f"{subject}")
    ax.legend(wedges, subject_data['Subject_Full_Text'], title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
