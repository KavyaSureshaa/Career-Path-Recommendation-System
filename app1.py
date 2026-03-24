import streamlit as st
import pandas as pd
import joblib
import re

# ==============================================
# LOAD MODELS
# ==============================================

@st.cache_resource
def load_models():
    model = joblib.load("model/svc_model.pkl")
    tfidf = joblib.load("model/tfidf.pkl")
    le    = joblib.load("model/label_encoder.pkl")
    df    = pd.read_csv("model/jobs_cleaned.csv")
    return model, tfidf, le, df

model, tfidf, le, df = load_models()

# ==============================================
# CAREER GROWTH PATHS
# ==============================================

GROWTH_PATHS = {
    "it software": [
        ("🌱 Entry Level",  "Junior Software Developer / Trainee Engineer"),
        ("💼 Mid Level",    "Software Engineer / Developer"),
        ("🚀 Senior Level", "Senior Software Engineer"),
        ("🎯 Leadership",   "Tech Lead / Engineering Manager"),
        ("👑 Expert",       "Principal Engineer / VP Engineering"),
    ],
    "data science analytics": [
        ("🌱 Entry Level",  "Data Science Trainee / Junior Data Analyst"),
        ("💼 Mid Level",    "Data Scientist / ML Engineer"),
        ("🚀 Senior Level", "Senior Data Scientist"),
        ("🎯 Leadership",   "Lead Data Scientist / AI Tech Lead"),
        ("👑 Expert",       "Principal Data Scientist / AI Architect"),
    ],
    "analytics bi": [
        ("🌱 Entry Level",  "Junior BI Analyst / Data Analyst Trainee"),
        ("💼 Mid Level",    "BI Developer / Data Analyst"),
        ("🚀 Senior Level", "Senior BI Developer / Senior Analyst"),
        ("🎯 Leadership",   "Analytics Manager / BI Lead"),
        ("👑 Expert",       "Director of Analytics / Chief Data Officer"),
    ],
    "teaching education training": [
        ("🌱 Entry Level",  "Junior Trainer / Teaching Assistant"),
        ("💼 Mid Level",    "Corporate Trainer / Educator"),
        ("🚀 Senior Level", "Senior Trainer / Curriculum Designer"),
        ("🎯 Leadership",   "Training Manager / Head of Learning"),
        ("👑 Expert",       "Director of Education / Chief Learning Officer"),
    ],
    "finance": [
        ("🌱 Entry Level",  "Junior Finance Analyst / Accounts Trainee"),
        ("💼 Mid Level",    "Finance Analyst / Accountant"),
        ("🚀 Senior Level", "Senior Finance Analyst / Finance Manager"),
        ("🎯 Leadership",   "Finance Controller / CFO"),
        ("👑 Expert",       "Chief Financial Officer / Finance Director"),
    ],
    "hr recruitment": [
        ("🌱 Entry Level",  "HR Trainee / Junior Recruiter"),
        ("💼 Mid Level",    "HR Executive / Recruiter"),
        ("🚀 Senior Level", "Senior HR Manager"),
        ("🎯 Leadership",   "HR Lead / HRBP"),
        ("👑 Expert",       "CHRO / Head of Human Resources"),
    ],
    "marketing advertising mr pr": [
        ("🌱 Entry Level",  "Marketing Trainee / Junior Executive"),
        ("💼 Mid Level",    "Marketing Executive / PR Executive"),
        ("🚀 Senior Level", "Senior Marketing Manager"),
        ("🎯 Leadership",   "Marketing Head / Brand Manager"),
        ("👑 Expert",       "Chief Marketing Officer"),
    ],
    "sales retail business development": [
        ("🌱 Entry Level",  "Sales Trainee / Junior BD Executive"),
        ("💼 Mid Level",    "Sales Executive / BD Manager"),
        ("🚀 Senior Level", "Senior Sales Manager"),
        ("🎯 Leadership",   "Regional Sales Head"),
        ("👑 Expert",       "VP Sales / Chief Revenue Officer"),
    ],
    "engineering design": [
        ("🌱 Entry Level",  "Junior Engineer / Design Trainee"),
        ("💼 Mid Level",    "Design Engineer"),
        ("🚀 Senior Level", "Senior Design Engineer"),
        ("🎯 Leadership",   "Lead Engineer / Engineering Manager"),
        ("👑 Expert",       "Principal Engineer / Chief Engineer"),
    ],
    "systems it infrastructure": [
        ("🌱 Entry Level",  "IT Support / Junior System Admin"),
        ("💼 Mid Level",    "System Administrator / Network Engineer"),
        ("🚀 Senior Level", "Senior Infrastructure Engineer"),
        ("🎯 Leadership",   "IT Manager / Infrastructure Lead"),
        ("👑 Expert",       "CTO / Head of IT Infrastructure"),
    ],
}

def get_growth_path(functional_area, base_label):
    key = functional_area.lower().strip()
    for path_key, steps in GROWTH_PATHS.items():
        if path_key in key or key in path_key:
            return steps
    return [
        ("🌱 Entry Level",  f"Junior {base_label} / Trainee"),
        ("💼 Mid Level",    f"{base_label}"),
        ("🚀 Senior Level", f"Senior {base_label}"),
        ("🎯 Leadership",   f"Lead / Manager — {base_label}"),
        ("👑 Expert",       f"Principal {base_label} / Director"),
    ]

# ==============================================
# CLEAN LABEL DISPLAY
# ==============================================

LABEL_DISPLAY = {
    "it software"                         : "IT Software",
    "data science analytics"              : "Data Science & Analytics",
    "analytics bi"                        : "Analytics & Business Intelligence",
    "teaching education training"         : "Teaching, Education & Training",
    "finance"                             : "Finance & Accounting",
    "hr recruitment"                      : "HR & Recruitment",
    "marketing advertising mr pr"         : "Marketing, Advertising & PR",
    "sales retail business development"   : "Sales & Business Development",
    "engineering design"                  : "Engineering & Design",
    "systems it infrastructure"           : "IT Infrastructure & Systems",
    "programming design"                  : "Software Programming & Design",
    "erp crm"                             : "ERP & CRM",
    "network system administration"       : "Network & System Administration",
    "it hardware"                         : "IT Hardware",
    "quality"                             : "Quality Assurance",
    "production manufacturing maintenance": "Production & Manufacturing",
    "health care"                         : "Healthcare",
    "banking insurance"                   : "Banking & Insurance",
    "legal"                               : "Legal & Compliance",
    "media arts"                          : "Media & Arts",
    "supply chain logistics"              : "Supply Chain & Logistics",
    "top management"                      : "Top Management",
    "project management"                  : "Project Management",
    "customer service"                    : "Customer Service & Support",
}

def get_display_label(raw_label):
    key = raw_label.lower().strip()
    return LABEL_DISPLAY.get(key, raw_label.title())

# ==============================================
# PAGE CONFIGURATION
# ==============================================

st.set_page_config(
    page_title="Career Path Recommendation",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.3rem;
        font-weight: 900;
        color: #1a237e;
        padding: 10px 0 4px 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .career-box {
        background: linear-gradient(135deg, #1a237e, #1565c0);
        color: white;
        padding: 28px;
        border-radius: 18px;
        text-align: center;
        font-size: 2rem;
        font-weight: 900;
        margin: 15px 0 20px 0;
    }
    .source-note {
        text-align: center;
        color: #888;
        font-size: 0.82rem;
        margin-bottom: 10px;
    }
    .growth-step {
        background: #f4f6ff;
        border-left: 5px solid #1a237e;
        padding: 13px 20px;
        margin: 7px 0;
        border-radius: 8px;
        font-size: 1rem;
    }
    .growth-current {
        background: #fff8e1;
        border-left: 5px solid #f9a825;
        padding: 13px 20px;
        margin: 7px 0;
        border-radius: 8px;
        font-size: 1rem;
    }
    .job-card {
        background: #ffffff;
        border: 1px solid #e0e4f0;
        padding: 10px 16px;
        border-radius: 8px;
        margin: 5px 0;
        font-size: 0.92rem;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎯 Career Path Recommendation System</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered system using LinearSVC + TF-IDF '
    'trained on real Naukri.com job market data</div>',
    unsafe_allow_html=True
)

# ==============================================
# ▼▼▼ SIDEBAR — THIS IS WHERE ANSWER 3 GOES ▼▼▼
# ==============================================

st.sidebar.header("👤 Enter Your Profile")

# ── SKILLS DROPDOWN ──────────────────────────
ALL_SKILLS = sorted([
    "Python", "Machine Learning", "Deep Learning", "SQL", "Data Analysis",
    "TensorFlow", "Keras", "PyTorch", "NLP", "Computer Vision",
    "Power BI", "Tableau", "Excel", "Data Visualization", "ETL",
    "Java", "C++", "JavaScript", "React", "Angular", "Node JS",
    "HTML", "CSS", "PHP", "Django", "Flask", "Spring Boot",
    "Android", "iOS", "Flutter", "React Native", "Swift",
    "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes",
    "DevOps", "Jenkins", "Terraform", "Linux", "Shell Scripting",
    "Cybersecurity", "Ethical Hacking", "Network Security",
    "Penetration Testing", "MySQL", "PostgreSQL", "MongoDB",
    "Oracle DBA", "Redis", "Selenium", "Manual Testing",
    "Test Automation", "JMeter", "SAP", "ERP", "SAP ABAP",
    "SAP HANA", "Project Management", "Agile", "Scrum", "JIRA",
    "Business Analysis", "UI Design", "UX Design", "Figma",
    "Adobe XD", "Embedded Systems", "IoT", "Arduino", "VLSI",
    "Digital Marketing", "SEO", "Content Writing", "Social Media",
    "Accounting", "Finance", "Tally", "GST", "Auditing",
    "HR Management", "Recruitment", "Payroll",
    "Teaching", "Curriculum Design", "E-Learning", "Coaching",
])

selected_skills = st.sidebar.multiselect(
    "💡 Select Your Skills",
    options=ALL_SKILLS,
    default=["Python", "Machine Learning", "SQL"]
)

extra_skills = st.sidebar.text_input(
    "➕ Add extra skills not in list above", ""
)

# Combine dropdown + typed skills
all_user_skills = selected_skills.copy()
if extra_skills.strip():
    extra_list = [
        s.strip() for s in
        extra_skills.replace(",", " ").split()
        if s.strip()
    ]
    all_user_skills += extra_list

if all_user_skills:
    st.sidebar.caption("✅ Skills selected: " + ", ".join(all_user_skills))
else:
    st.sidebar.warning("⚠️ Please select at least one skill.")

# ── OTHER INPUTS ──────────────────────────────
experience = st.sidebar.slider("📅 Years of Experience", 0, 20, 2)

education = st.sidebar.selectbox(
    "🎓 Education Level",
    ["B.Tech", "M.Tech", "BSc", "MSc", "PhD", "MBA", "BCA", "MCA", "Diploma"]
)

recommend_btn = st.sidebar.button(
    "🔍 Recommend My Career Path",
    use_container_width=True
)

# ==============================================
# ▲▲▲ SIDEBAR ENDS HERE ▲▲▲
# ==============================================

# METRICS ROW
c1, c2, c3, c4 = st.columns(4)
c1.metric("🤖 Model",      "LinearSVC")
c2.metric("📂 Dataset",    f"{len(df):,} jobs")
c3.metric("🏷️ Categories", f"{df['Functional Area'].nunique()}")
c4.metric("📐 Features",   "TF-IDF (8000)")

st.divider()

# ==============================================
# RECOMMENDATION ENGINE
# ==============================================
if recommend_btn:

    if not all_user_skills:
        st.warning("⚠️ Please select at least one skill from the dropdown.")
        st.stop()

    # Build skills string from dropdown + extra input
    skills_clean = " ".join(all_user_skills).lower()
    skills_clean = re.sub(r'[^a-zA-Z ]', ' ', skills_clean)
    skills_clean = re.sub(r'\s+', ' ', skills_clean).strip()

    # For summary display
    skills_display = ", ".join(all_user_skills)

    user_profile = skills_clean + " " + education.lower()

    # Vectorize using trained TF-IDF
    input_vec = tfidf.transform([user_profile])

    # Predict using LinearSVC
    predicted_label   = model.predict(input_vec)[0]
    predicted_raw     = le.inverse_transform([predicted_label])[0]
    predicted_display = get_display_label(predicted_raw)

    # SHOW PREDICTED CAREER
    st.markdown(
        "<h2 style='text-align:center; color:#1a237e;'>"
        "✅ Your Recommended Career Domain</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div class='career-box'>🏆 {predicted_display}</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='source-note'>📌 Label source: <b>Functional Area</b> column "
        "from Naukri.com dataset — predicted by LinearSVC trained on real job market data.</div>",
        unsafe_allow_html=True
    )

    st.divider()

    # MATCHING JOB TITLES
    st.subheader("📌 Matching Job Titles From Dataset")

    matched = df[
        df['Functional Area'].str.lower().str.strip() == predicted_raw.lower().strip()
    ]['Job Title'].drop_duplicates()

    if len(matched) == 0:
        matched = df[
            df['Functional Area'].str.lower().str.contains(
                predicted_raw.lower().split()[0], na=False
            )
        ]['Job Title'].drop_duplicates()

    matched = matched.head(10).tolist()

    if matched:
        col1, col2 = st.columns(2)
        for i, job in enumerate(matched):
            card = f"<div class='job-card'>💼 {job.title()}</div>"
            if i % 2 == 0:
                col1.markdown(card, unsafe_allow_html=True)
            else:
                col2.markdown(card, unsafe_allow_html=True)
    else:
        st.info("No matching job titles found in dataset for this category.")

    st.divider()

    # CAREER GROWTH PATH
    st.subheader("📈 Your Personalized Career Growth Path")

    path = get_growth_path(predicted_raw, predicted_display)

    if experience <= 1:
        current_idx = 0
    elif experience <= 3:
        current_idx = 1
    elif experience <= 6:
        current_idx = 2
    elif experience <= 10:
        current_idx = 3
    else:
        current_idx = 4

    for i, (level, title) in enumerate(path):
        if i == current_idx:
            st.markdown(
                f"<div class='growth-current'>"
                f"<b>{level}</b> &nbsp;➡️&nbsp; {title} &nbsp;"
                f"<span style='background:#f9a825; color:white; "
                f"padding:2px 10px; border-radius:12px; "
                f"font-size:0.78rem;'>📍 You are here</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='growth-step'>"
                f"<b>{level}</b> &nbsp;➡️&nbsp; {title}"
                f"</div>",
                unsafe_allow_html=True
            )

    st.divider()

    # INPUT SUMMARY
    st.subheader("📝 Your Input Summary")

    s1, s2 = st.columns(2)
    with s1:
        st.write("**🔧 Skills :**",    skills_display)
        st.write("**🎓 Education :**", education)
    with s2:
        st.write("**📅 Experience :**", f"{experience} year(s)")

    st.success(
        f"✅ Best career domain for your profile: **{predicted_display}**"
    )

# FOOTER
st.divider()
st.markdown(
    "<center><small>🎯 Career Path Recommendation System &nbsp;|&nbsp; "
    "LinearSVC + TF-IDF &nbsp;|&nbsp; "
    "Labels from Naukri.com Dataset Functional Area</small></center>",
    unsafe_allow_html=True
)
