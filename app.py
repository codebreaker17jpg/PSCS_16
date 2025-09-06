import streamlit as st
import pandas as pd
from src.recommender import recommend_jobs
import os

# Load data
users = pd.read_csv("data/punjab_users.csv")

st.title("Punjab Job Analytics & Recommender System")

# Analytics section
st.header("📊 User Analytics")

st.write("### Education Distribution")
st.image("docs/edu_dist.png")

st.write("### Gender Split")
st.image("docs/gender_split.png")

st.write("### Users by Location")
st.image("docs/users_by_location.png")

st.write("### Application Status")
st.image("docs/app_status.png")

# Recommender section
st.header("🤝 Job Recommendations")

user_id = st.selectbox("Select a user ID", users["user_id"].unique())
if st.button("Recommend Jobs"):
    recs = recommend_jobs(user_id, top_n=5)
    st.write("### Top Job Recommendations")
    st.dataframe(recs)
