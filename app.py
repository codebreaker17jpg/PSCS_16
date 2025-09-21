# app.py - polished analytics + recommender demo
import streamlit as st
import pandas as pd
import time
from sqlalchemy import create_engine
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import altair as alt
from collections import Counter
import io

# try to import AgGrid (optional)
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode
    AGGRID_AVAILABLE = True
except Exception:
    AGGRID_AVAILABLE = False

# ---------- CONFIG ----------
DB_URL = "postgresql://postgres:pass@localhost:5432/pscs16"  # change if needed
engine = create_engine(DB_URL)
st.set_page_config(page_title="PGRKAM Analytics & Recommender", layout="wide")

# ---------- Data loader with caching ----------
@st.cache_data(ttl=300)
def load_tables():
    users = pd.read_sql("SELECT * FROM users", engine)
    jobs  = pd.read_sql("SELECT * FROM jobs", engine)
    apps  = pd.read_sql("SELECT * FROM applications", engine)
    return users, jobs, apps

# ---------- helper utilities ----------
def compute_matched_skills(user_skills, job_skills_str):
    try:
        uj = set([s.strip().lower() for s in str(user_skills).split(";") if s.strip()])
        jk = set([s.strip().lower() for s in str(job_skills_str).split(";") if s.strip()])
        common = sorted(list(uj & jk))
        return ";".join(common) if common else ""
    except Exception:
        return ""

def download_figure_bytes(fig, fmt="png"):
    # plotly figure -> bytes (requires kaleido installed for png/svg)
    buf = io.BytesIO()
    fig.write_image(buf, format=fmt)
    buf.seek(0)
    return buf

# ---------- layout ----------
st.title("PGRKAM — Analytics & Job Recommender (Polished Demo)")

# Load data
with st.spinner("Loading data from DB..."):
    try:
        users_df, jobs_df, apps_df = load_tables()
    except Exception as e:
        st.error("Failed to load data from DB. Check DB_URL and that the database is up.")
        st.exception(e)
        st.stop()

# KPI row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Users", f"{len(users_df):,}")
col2.metric("Jobs", f"{len(jobs_df):,}")
col3.metric("Applications", f"{len(apps_df):,}")
selected_count = (apps_df['status']=='Selected').sum() if 'status' in apps_df.columns else 0
succ_rate = (selected_count / len(apps_df) * 100) if len(apps_df)>0 else 0.0
col4.metric("Success rate", f"{succ_rate:.2f}%")

st.markdown("---")

# Tabs for analytics and recommender
tab_analytics, tab_recommender = st.tabs(["Analytics", "Recommender"])

# ---------------- Analytics Tab ----------------
with tab_analytics:
    st.header("Analytics Dashboard")

    # Filters
    with st.expander("Filters", expanded=False):
        filt_col1, filt_col2 = st.columns(2)
        unique_locations = sorted(users_df['location'].dropna().unique().tolist())
        sel_locations = filt_col1.multiselect("Filter by user location", options=unique_locations, default=None)
        top_titles = jobs_df['title'].value_counts().nlargest(10).index.tolist()
        sel_titles = filt_col2.multiselect("Filter by job title (top 10)", options=top_titles, default=None)

    # apply filters
    users_filt = users_df.copy()
    jobs_filt = jobs_df.copy()
    apps_filt = apps_df.copy()
    if sel_locations:
        users_filt = users_filt[users_filt['location'].isin(sel_locations)]
        apps_filt = apps_filt.merge(users_filt[['user_id']], on='user_id', how='inner')
    if sel_titles:
        jobs_filt = jobs_filt[jobs_filt['title'].isin(sel_titles)]

    # Row: plotly charts
    fig_col1, fig_col2 = st.columns([2,1])
    # Users by location (Plotly)
    loc_counts = users_filt['location'].value_counts().reset_index()
    loc_counts.columns = ['location','count']
    fig_loc = px.bar(loc_counts.sort_values('count'), x='count', y='location', orientation='h',
                     title="Users by Location", labels={'count':'# Users','location':''})
    fig_col1.plotly_chart(fig_loc, use_container_width=True)

    # Application status pie
    status_counts = apps_filt['status'].value_counts().reset_index()
    status_counts.columns = ['status','count']
    fig_status = px.pie(status_counts, names='status', values='count', title="Application Status")
    fig_col2.plotly_chart(fig_status, use_container_width=True)

    # Funnel
    st.subheader("User Funnel")
    funnel_df = pd.DataFrame({
        'stage': ['Users','Applied','Selected'],
        'count': [len(users_df), len(apps_df), selected_count]
    })
    fig_funnel = px.funnel(funnel_df, x='count', y='stage', title='User Funnel')
    st.plotly_chart(fig_funnel, use_container_width=True)

    # WordCloud of skills
    st.subheader("Top Skills WordCloud")
    skills_series = users_df['skills'].dropna().str.split(";").explode().str.strip().str.lower()
    freq = Counter(skills_series)
    wc = WordCloud(width=900, height=300, background_color="white").generate_from_frequencies(freq)
    fig_wc = plt.figure(figsize=(12,4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig_wc)
    plt.close(fig_wc)

    # Heatmap: location x top titles (Altair)
    st.subheader("Jobs Heatmap (Location × Top Titles)")
    top_titles2 = jobs_df['title'].value_counts().nlargest(10).index.tolist()
    sub = jobs_df[jobs_df['title'].isin(top_titles2)]
    pivot = sub.groupby(['location','title']).size().reset_index(name='count')
    heat = alt.Chart(pivot).mark_rect().encode(
        x=alt.X('title:N', sort=top_titles2, title='Job Title'),
        y=alt.Y('location:N', title='Location'),
        color=alt.Color('count:Q', title='Number of Jobs'),
        tooltip=['location','title','count']
    ).properties(width='container', height=350)
    st.altair_chart(heat, use_container_width=True)

    # Optionally show raw tables (collapsible)
    with st.expander("Preview: Users / Jobs / Applications (sample)", expanded=False):
        st.write("Users (sample 200):")
        st.dataframe(users_df.sample(200).reset_index(drop=True))
        st.write("Jobs (sample 200):")
        st.dataframe(jobs_df.sample(min(200,len(jobs_df))).reset_index(drop=True))
        st.write("Applications (sample 200):")
        st.dataframe(apps_df.sample(min(200,len(apps_df))).reset_index(drop=True))

# ---------------- Recommender Tab ----------------
with tab_recommender:
    st.header("Job Recommender")

    # Recommender inputs
    col_left, col_right = st.columns([2,1])
    with col_left:
        typed_user = st.text_input("Enter User ID (1–50000):", "")
        if typed_user.strip():
            try:
                uid = int(typed_user.strip())
            except:
                st.error("Please enter a valid integer user id")
                uid = None
        else:
            uid = None

    with col_right:
        top_n = st.number_input("Number of recommendations", min_value=1, max_value=20, value=5, step=1)
        same_city_boost = st.checkbox("Boost same city jobs", value=True)
        run_button = st.button("Recommend Jobs")

    # When clicked -> run recommender (with spinner + placeholder)
    if run_button:
        if uid is None:
            st.warning("Please enter a valid user id.")
        else:
            status = st.empty()
            status.info(f"Generating recommendations for user {uid} ...")
            with st.spinner("Computing recommendations..."):
                try:
                    # import here to ensure src package path is resolved (src must exist)
                    from src.recommender import recommend_jobs
                    recs = recommend_jobs(uid, top_n=top_n, same_city_boost=same_city_boost)
                except Exception as e:
                    status.empty()
                    st.error("Error executing recommender.")
                    st.exception(e)
                    recs = None

            # clear status
            status.empty()

            if recs is None or recs.empty:
                st.warning("No recommendations returned for this user.")
            else:
                # enrich recs with matched skills
                user_row = users_df[users_df['user_id']==uid]
                user_sk = user_row['skills'].iloc[0] if not user_row.empty else ""
                recs = recs.copy()
                recs['matched_skills'] = recs['skills_required'].apply(lambda s: compute_matched_skills(user_sk, s))
                if 'score' in recs.columns:
                    recs['score'] = recs['score'].astype(float).round(3)

                st.subheader(f"Top {len(recs)} recommendations for user {uid} (computed in approx realtime)")
                # interactive grid if available
                display_df = recs[['job_id','title','location','matched_skills','score']].reset_index(drop=True)
                if AGGRID_AVAILABLE:
                    gb = GridOptionsBuilder.from_dataframe(display_df)
                    gb.configure_default_column(filterable=True, sortable=True, resizable=True)
                    gb.configure_selection(selection_mode="single", use_checkbox=False)
                    gridOptions = gb.build()
                    AgGrid(display_df, gridOptions=gridOptions, height=300, fit_columns_on_grid_load=True)
                else:
                    st.dataframe(display_df, use_container_width=True)

                st.markdown("**Explanation:** Recommendations are primarily based on skill similarity (cosine similarity on skill tokens), with optional city and popularity boosts.")
                # allow download of the results CSV
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download recommendations (CSV)", data=csv, file_name=f"recs_user_{uid}.csv", mime="text/csv")

# Footer
st.markdown("---")
st.caption("Demo prepared for Review-3. Data is synthetic for demonstration purposes.")
