import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load the datasets
try:
    users_df = pd.read_csv('users_50k.csv')
    jobs_df = pd.read_csv('jobs_2k.csv')
    apps_df = pd.read_csv('applications_300k.csv')
    print("CSV files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Please place the CSV files in the same directory as this script.")
    exit()

# --- 1. User Demographics ---

# User Distribution by Location
plt.figure(figsize=(12, 7))
location_counts = users_df['location'].value_counts()
sns.barplot(x=location_counts.values, y=location_counts.index, palette='viridis')
plt.title('Figure 1: User Distribution by Location', fontsize=16)
plt.xlabel('Number of Users', fontsize=12)
plt.ylabel('Location', fontsize=12)
plt.tight_layout()
plt.savefig('user_distribution_location.png')
print("Generated user_distribution_location.png")

# User Acquisition Channels
plt.figure(figsize=(10, 10))
channel_counts = users_df['channel'].value_counts()
plt.pie(channel_counts, labels=channel_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Figure 2: User Acquisition Channels', fontsize=16)
plt.ylabel('')
plt.tight_layout()
plt.savefig('user_acquisition_channels.png')
print("Generated user_acquisition_channels.png")

# User Education Levels
plt.figure(figsize=(12, 7))
education_counts = users_df['education'].value_counts()
sns.barplot(x=education_counts.values, y=education_counts.index, palette='plasma')
plt.title('Figure 3: User Education Levels', fontsize=16)
plt.xlabel('Number of Users', fontsize=12)
plt.ylabel('Education', fontsize=12)
plt.tight_layout()
plt.savefig('user_education_levels.png')
print("Generated user_education_levels.png")

# --- 2. Skills Analysis ---

# User Skills Analysis
user_skills_list = ';'.join(users_df['skills'].dropna()).split(';')
user_skills_counts = Counter(s.strip() for s in user_skills_list if s.strip())
top_user_skills = pd.DataFrame(user_skills_counts.most_common(15), columns=['Skill', 'Count'])

plt.figure(figsize=(12, 8))
sns.barplot(x='Count', y='Skill', data=top_user_skills, palette='coolwarm')
plt.title('Figure 4: Top 15 Skills Among Users', fontsize=16)
plt.xlabel('Number of Users', fontsize=12)
plt.ylabel('Skill', fontsize=12)
plt.tight_layout()
plt.savefig('top_user_skills.png')
print("Generated top_user_skills.png")

# Job Skills Analysis
job_skills_list = ';'.join(jobs_df['skills_required'].dropna()).split(';')
job_skills_counts = Counter(s.strip() for s in job_skills_list if s.strip())
top_job_skills = pd.DataFrame(job_skills_counts.most_common(15), columns=['Skill', 'Count'])

plt.figure(figsize=(12, 8))
sns.barplot(x='Count', y='Skill', data=top_job_skills, palette='magma')
plt.title('Figure 5: Top 15 Skills Required in Jobs', fontsize=16)
plt.xlabel('Number of Job Postings', fontsize=12)
plt.ylabel('Skill', fontsize=12)
plt.tight_layout()
plt.savefig('top_job_skills.png')
print("Generated top_job_skills.png")

# --- 3. Application Status Analysis ---

plt.figure(figsize=(8, 8))
status_counts = apps_df['status'].value_counts()
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Figure 6: Job Application Status', fontsize=16)
plt.ylabel('')
plt.tight_layout()
plt.savefig('application_status_pie_chart.png')
print("Generated application_status_pie_chart.png")