# tools/generate_synthetic.py
import random, csv, os
from faker import Faker
from tqdm import trange
import pandas as pd
fake = Faker()

OUT = "data_synthetic"
os.makedirs(OUT, exist_ok=True)

# small vocab
skills_vocab = ["python","java","sql","excel","ml","dl","nlp","django","flask","aws","docker","kubernetes",
                "sales","marketing","accounting","hr","communication","project_management","c++","react","nodejs"]

locations = ["Amritsar","Ludhiana","Jalandhar","Chandigarh","Patiala","Mohali","Hoshiarpur","Bathinda","Firozpur"]
educations = ["High School","Diploma","BSc","BCom","BA","BTech","MSc","MBA"]

def make_skills(min_k=1, max_k=6):
    k = random.randint(min_k, max_k)
    return random.sample(skills_vocab, k)

N_USERS = 50000
N_JOBS = 2000
# make users CSV
users_file = os.path.join(OUT,"users_50k.csv")
with open(users_file, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["user_id","name","age","gender","education","location","skills","channel"])
    for i in trange(N_USERS):
        uid = i+1
        name = fake.name()
        age = random.randint(18,55)
        gender = random.choice(["Male","Female","Other"])
        edu = random.choice(educations)
        loc = random.choice(locations)
        skills = ";".join(make_skills())
        channel = random.choice(["Facebook","WhatsApp","Newspaper","Govt_Website","WordOfMouth","Referral"])
        writer.writerow([uid,name,age,gender,edu,loc,skills,channel])

# make jobs CSV
jobs_file = os.path.join(OUT,"jobs_2k.csv")
with open(jobs_file,"w",newline='',encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["job_id","title","company","skills_required","location"])
    titles = ["Data Analyst","Software Developer","Sales Executive","Accountant","HR Executive","Field Officer",
              "Full Stack Developer","Machine Learning Engineer","Quality Analyst","Digital Marketer"]
    for j in trange(N_JOBS):
        jid = j+1
        title = random.choice(titles)
        company = fake.company()
        skills = ";".join(make_skills(2,8))
        loc = random.choice(locations)
        writer.writerow([jid,title,company,skills,loc])

# make applications CSV - simulate interactions: avg 3 apps/user
apps_file = os.path.join(OUT,"applications_300k.csv")
with open(apps_file,"w",newline='',encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["id","user_id","job_id","status"])
    idcnt = 1
    for uid in trange(1, N_USERS+1):
        # number of apps per user - skewed distribution
        n_apps = max(1, int(random.expovariate(1/2)))  # avg ~2
        for _ in range(n_apps):
            job_id = random.randint(1, N_JOBS)
            # simulate success probability higher if skill overlap (approx)
            # quick hack: compute overlap by sampling skills roughly
            # We'll randomly decide status with some bias
            prob_selected = 0.05  # base
            # bump if location likely matched etc (just simulate)
            if random.random() < 0.15:
                prob_selected += 0.1
            status = "Selected" if random.random() < prob_selected else "Rejected"
            writer.writerow([idcnt, uid, job_id, status])
            idcnt += 1

print("Generated files in", OUT)
