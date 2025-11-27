# modules
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
import plotly.express as px

# set a light template and matching color sequence
px.defaults.template = "simple_white"
px.defaults.color_discrete_sequence = ['#FF6B6B', '#FF8A5B', '#FFD166', '#8BD3C7', '#5A8BD3']

# try to import aggrid (optional)
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode
    AGGRID_AVAILABLE = True
except Exception:
    AGGRID_AVAILABLE = False

# config 
DB_URL = "postgresql://postgres:pass@localhost:5432/pscs16"  
engine = create_engine(DB_URL)
st.set_page_config(page_title="PGRKAM Analytics & Recommender", layout="wide")
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# Header
def render_header(title="Analytics & Job Recommender", subtitle="Discover Oppurtunities"):
    hdr = f"""
    <div class="header">
      <div class="header-logo">PGRKAM</div>
      <div style="line-height:1;">
        <div style="font-size:20px;font-weight:700;color:var(--indigo);">{title}</div>
        <div style="font-size:13px;color:var(--muted);margin-top:3px;">{subtitle}</div>
      </div>
      <div style="margin-left:auto;">
        <div class="small-muted">Demo • Analytics • Recommender</div>
      </div>
    </div>
    """
    st.markdown(hdr, unsafe_allow_html=True)

# KPI card renderer
def kpi_card(title, value, help_text=""):
    html = f"""
    <div class="kpi-card">
      <div class="kpi-title">{title}</div>
      <div class="kpi-value">{value}</div>
      <div class="small-muted" style="margin-top:6px;">{help_text}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


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

<<<<<<< HEAD
# ---------- layout ----------
st.title("PGRKAM — Analytics & Job Recommender")
=======
# layout 
st.title("Punjab Ghar ghar Rozgar and Karobar Mission")
>>>>>>> dca20f8 (style.css)

# Load data
with st.spinner("Loading data from DB..."):
    try:
        users_df, jobs_df, apps_df = load_tables()
    except Exception as e:
        st.error("Failed to load data from DB. Check DB_URL and that the database is up.")
        st.exception(e)
        st.stop()

# KPI row
render_header()  # replace st header 

selected_count = (apps_df['status'] == 'Selected').sum() if 'status' in apps_df.columns else 0
succ_rate = (selected_count / len(apps_df) * 100) if len(apps_df) > 0 else 0.0

kcols = st.columns(4)
kpi_vals = [
    ("Users", f"{len(users_df):,}", "Total registered users"),
    ("Jobs", f"{len(jobs_df):,}", "Open job postings"),
    ("Applications", f"{len(apps_df):,}", "Total applications submitted"),
    ("Success rate", f"{succ_rate:.2f}%", "Selected / Applications")
]
for col, (t, v, desc) in zip(kcols, kpi_vals):
    with col:
        kpi_card(t, v, desc)

st.markdown("---")

# Tabs for analytics and recommender
tab_analytics, tab_recommender = st.tabs(["Analytics", "Recommender"])

# Analytics Tab 
with tab_analytics:
    st.header("Analytics Dashboard")

    # Filters
    with st.expander("Filters", expanded=False):
        filt_col1, filt_col2 = st.columns(2)
        unique_locations = sorted(users_df['location'].dropna().unique().tolist())
        sel_locations = filt_col1.multiselect("Filter by user location", options=unique_locations, default=[])
        top_titles = jobs_df['title'].value_counts().nlargest(10).index.tolist()
        sel_titles = filt_col2.multiselect("Filter by job title (top 10)", options=top_titles, default=[])

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
    loc_counts = users_filt['location'].value_counts().reset_index() if not users_filt.empty else pd.DataFrame(columns=['index','location'])
    if not loc_counts.empty:
     loc_counts.columns = ['location','count']
    else:
     loc_counts = pd.DataFrame({'location':[], 'count':[]})
    fig_loc = px.bar(loc_counts.sort_values('count'), x='count', y='location', orientation='h',
                 title="Users by Location", labels={'count':'# Users','location':''})

    fig_col1.plotly_chart(fig_loc, use_container_width=True)

    # Application status pie
    
    if sel_titles:
       apps_filt = apps_filt.merge(jobs_filt[['job_id']], on='job_id', how='inner')
    status_counts = apps_filt['status'].value_counts().reset_index()
    status_counts.columns = ['status','count']
    fig_status = px.pie(status_counts, names='status', values='count', title="Application Status")

    fig_col2.plotly_chart(fig_status, use_container_width=True)

    # Funnel (already created above)
    st.subheader("User Funnel")
    funnel_df = pd.DataFrame({
        'stage': ['Users','Applied','Selected'],
        'count': [len(users_df), len(apps_df), selected_count]
    })
    fig_funnel = px.funnel(funnel_df, x='count', y='stage', title='User Funnel')
    st.plotly_chart(fig_funnel, use_container_width=True)

    # WordCloud
    st.subheader("Top Skills WordCloud")

    # Choice of source
    wc_mode = st.radio("WordCloud source:", options=["All users", "Jobs (select job)", "User vs Job matched"], index=0)

    def split_skills(cell, delimiter=";"):
        return [s.strip().lower() for s in str(cell).split(delimiter) if s.strip()]

    from collections import Counter
    import numpy as np
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    if wc_mode == "All users":
        skills_series = users_df['skills'].dropna().str.split(";").explode().str.strip().str.lower()
        freq = Counter(skills_series)

    elif wc_mode == "Jobs (select job)":
        job_sel_by = st.selectbox("Select job selector:", options=["Title","Job ID"])
        if job_sel_by == "Title":
            job_title = st.selectbox("Choose job title:", options=sorted(jobs_df['title'].unique().tolist()))
            sub_jobs = jobs_df[jobs_df['title'] == job_title]
            if len(sub_jobs) > 1:
                jid = st.selectbox("Choose job posting (job_id):", options=sub_jobs['job_id'].tolist())
                skills_list = split_skills(jobs_df.loc[jobs_df['job_id']==jid, 'skills_required'].iloc[0])
            else:
                skills_list = split_skills(sub_jobs['skills_required'].iloc[0])
        else:
            jids = jobs_df['job_id'].tolist()
            jid = st.selectbox("Choose job id:", options=jids)
            skills_list = split_skills(jobs_df.loc[jobs_df['job_id']==jid, 'skills_required'].iloc[0])
        freq = Counter(skills_list)

    else:  # User vs Job matched
        uid_input = st.text_input("Enter user id (for matching):", "")
        job_choice = st.selectbox("Choose job for comparison (title):", options=sorted(jobs_df['title'].unique().tolist()))
        job_row = jobs_df[jobs_df['title'] == job_choice].iloc[0]
        job_skills = split_skills(job_row['skills_required'])
        try:
            uid = int(uid_input) if uid_input.strip() else None
        except:
            uid = None

        user_skills = []
        if uid is not None:
            urow = users_df[users_df['user_id'] == uid]
            if not urow.empty:
                user_skills = split_skills(urow['skills'].iloc[0])

        matched = set([s for s in user_skills if s in job_skills])
        job_only = [s for s in job_skills if s not in matched]
        user_only = [s for s in user_skills if s not in matched]

        freq = Counter()
        for s in matched:
            freq[s] += 50
        for s in job_only:
            freq[s] += 20
        for s in user_only:
            freq[s] += 5

    if not freq:
        st.info("No skills found for the selected source. Make sure the dataset columns are populated.")
    else:
        # ensure previous matplotlib figures are closed (prevents duplicates)
        plt.close("all")

        wc = WordCloud(
            width=900,
            height=300,
            background_color="white",
            max_words=100,
            prefer_horizontal=0.6,
            collocations=False,
        ).generate_from_frequencies(freq)

        # compute recolor sets safely
        matched_lower = set([m.lower() for m in matched]) if 'matched' in locals() else set()
        job_only_lower = set([j.lower() for j in job_only]) if 'job_only' in locals() else (set([s.lower() for s in skills_list]) if 'skills_list' in locals() else set())

        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            wl = word.lower()
            if wl in matched_lower:
                return "rgb(255,107,107)"   # coral
            elif wl in job_only_lower:
                return "rgb(255,209,102)"   # yellow
            else:
                return "rgb(120,120,120)"   # gray

        wc = wc.recolor(color_func=color_func)

        # display once, wrapped in styled div
        fig_wc = plt.figure(figsize=(12,4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.markdown('<div class="wordcloud-wrap">', unsafe_allow_html=True)
        st.pyplot(fig_wc)
        st.markdown('</div>', unsafe_allow_html=True)
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

# Recommender Tab 
with tab_recommender:
    st.header("Job Recommender")

    # inputs
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

    # When clicked it should run recommender 
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
                # Result download
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download recommendations (CSV)", data=csv, file_name=f"recs_user_{uid}.csv", mime="text/csv")

# Footer
st.markdown("---")
st.caption("Trying to make punjab better every step.")
