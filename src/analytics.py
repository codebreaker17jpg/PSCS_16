import os
import pandas as pd
import matplotlib.pyplot as plt

# Make sure docs folder exists for charts
os.makedirs("docs", exist_ok=True)

# Load datapyth
users = pd.read_csv("data/punjab_users.csv")
apps  = pd.read_csv("data/punjab_applications.csv")

# 1) Education distribution
ax = users["education"].value_counts().plot(kind="bar", title="Education Distribution")
plt.tight_layout(); plt.savefig("docs/edu_dist.png"); plt.close()

# 2) Gender distribution
ax = users["gender"].value_counts().plot(kind="pie", autopct="%1.0f%%", title="Gender Split")
plt.ylabel(""); plt.tight_layout(); plt.savefig("docs/gender_split.png"); plt.close()

# 3) Location-wise users
ax = users["location"].value_counts().plot(kind="bar", title="Users by Location")
plt.tight_layout(); plt.savefig("docs/users_by_location.png"); plt.close()

# 4) Application status
ax = apps["status"].value_counts().plot(kind="bar", title="Application Status Counts")
plt.tight_layout(); plt.savefig("docs/app_status.png"); plt.close()

print("✅ Charts saved inside docs/: edu_dist.png, gender_split.png, users_by_location.png, app_status.png")
