# tools/csv_to_postgres.py
import os
import pandas as pd
from sqlalchemy import create_engine, text
import argparse
import sys

def load_csv_to_postgres(df, table_name, engine, if_exists='replace'):
    print(f"Writing {len(df):,} rows -> {table_name} (if_exists={if_exists})")
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    print(f"Done: {table_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL", "postgresql://postgres:pass@localhost:5432/pscs16"))
    parser.add_argument("--data-dir", default="data_synthetic")
    parser.add_argument("--users-file", default="users_50k.csv")
    parser.add_argument("--jobs-file", default="jobs_2k.csv")
    parser.add_argument("--apps-file", default="applications_300k.csv")
    args = parser.parse_args()

    engine = create_engine(args.db_url)

    users_csv = os.path.join(args.data_dir, args.users_file)
    jobs_csv  = os.path.join(args.data_dir, args.jobs_file)
    apps_csv  = os.path.join(args.data_dir, args.apps_file)

    # sanity check files
    for p in (users_csv, jobs_csv, apps_csv):
        if not os.path.exists(p):
            print(f"ERROR: required file not found: {p}", file=sys.stderr)
            sys.exit(1)

    print("Connecting to DB:", args.db_url)

    # read CSVs into pandas (can be memory heavy for huge files; this is fine for your sizes)
    print("Reading CSVs into memory...")
    users_df = pd.read_csv(users_csv)
    jobs_df  = pd.read_csv(jobs_csv)
    apps_df  = pd.read_csv(apps_csv)

    # Basic cleaning (strip strings)
    for df in (users_df, jobs_df, apps_df):
        for c in df.select_dtypes(include=['object']).columns:
            df[c] = df[c].astype(str).str.strip()

    # Safe reload: drop dependent tables first, then reload in order
    with engine.begin() as conn:
        print("Dropping dependent tables (if exist) in safe order...")
        conn.execute(text("DROP TABLE IF EXISTS applications"))
        conn.execute(text("DROP TABLE IF EXISTS jobs"))
        conn.execute(text("DROP TABLE IF EXISTS users"))

    # Write tables in order
    load_csv_to_postgres(users_df, "users", engine, if_exists='replace')
    load_csv_to_postgres(jobs_df, "jobs", engine, if_exists='replace')

    # applications may refer to users/jobs by id; load it last
    load_csv_to_postgres(apps_df, "applications", engine, if_exists='replace')

    # Create indexes to speed up queries
    print("Creating indexes...")
    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_apps_user ON applications(user_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_apps_job ON applications(job_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_users_location ON users(location)"))
        # if you later convert skills to arrays/jsonb, add GIN indexes then

    print("All done. Tables created: users, jobs, applications.")
    print("You can verify counts by running SELECT COUNT(*) queries or using the verification commands in README.")

if __name__ == "__main__":
    main()
