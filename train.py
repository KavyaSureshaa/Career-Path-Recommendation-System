import pandas as pd
import re
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# ==============================================
# STEP 1: LOAD DATASET
# ==============================================

print("=" * 55)
print("  STEP 1: LOADING DATASET")
print("=" * 55)

df = pd.read_csv("E:\\Path recom\\data\\naukri_com-jobs.csv")

print("\nInitial Dataset Shape     :", df.shape)
print("Total Rows                :", df.shape[0])
print("Total Columns             :", df.shape[1])
print(df.head())
# Show all 11 columns with justification
print("\nAll Columns in Dataset:")
print(f"{'No.':<5} {'Column Name':<35} {'Selected?':<10} Reason")
print("-" * 80)

column_info = {
    'Job Title'                : ('YES', 'Directly describes the role — core feature'),
    'Key Skills'               : ('YES', 'Most important — skills define career domain'),
    'Job Experience Required'  : ('YES', 'Experience level affects career category'),
    'Role Category'            : ('YES', 'Helps model understand job domain'),
    'Industry'                 : ('YES', 'Industry context improves prediction'),
    'Functional Area'          : ('YES', 'TARGET LABEL — what we want to predict'),
    'Job Salary Offered'       : ('NO',  'Not relevant to career path prediction'),
    'Job Location(s)'          : ('NO',  'Location does not define career domain'),
    'Crawl Timestamp'          : ('NO',  'System metadata — no prediction value'),
    'Uniq Id'                  : ('NO',  'Just a row ID — no information'),
    'Job Description'          : ('NO',  'Too long, noisy — Key Skills already covers it'),
}

for i, (col, (selected, reason)) in enumerate(column_info.items(), 1):
    print(f"{i:<5} {col:<35} {selected:<10} {reason}")
df = df[['Job Title', 'Key Skills', 'Job Experience Required',
         'Role Category', 'Industry', 'Functional Area']]

print("Columns Selected          :", list(df.columns))

# ==============================================
# STEP 2: REMOVE MISSING VALUES
# ==============================================

print("\n" + "=" * 55)
print("  STEP 2: REMOVING MISSING VALUES")
print("=" * 55)

print("\nNull Values Per Column:")
print(df.isnull().sum().to_string())

rows_before = df.shape[0]
df = df.dropna()
rows_after  = df.shape[0]

print(f"\nRows Before : {rows_before}")
print(f"Rows After  : {rows_after}")
print(f"Rows Removed: {rows_before - rows_after}")

# ==============================================
# STEP 3: REMOVE DUPLICATES
# ==============================================

print("\n" + "=" * 55)
print("  STEP 3: REMOVING DUPLICATES")
print("=" * 55)

duplicates  = df.duplicated().sum()
rows_before = df.shape[0]
df          = df.drop_duplicates()
rows_after  = df.shape[0]

print(f"\nDuplicate Rows Found  : {duplicates}")
print(f"Rows Before           : {rows_before}")
print(f"Rows After            : {rows_after}")
print(f"Rows Removed          : {rows_before - rows_after}")

# ==============================================
# STEP 4: TEXT CLEANING
# ==============================================

print("\n" + "=" * 55)
print("  STEP 4: TEXT CLEANING")
print("=" * 55)

def clean_text(text):
    text = str(text).lower()               # lowercase
    text = re.sub(r'[^a-zA-Z ]', ' ', text)  # remove special chars
    text = re.sub(r'\s+', ' ', text)       # remove extra spaces
    return text.strip()

print("\nBefore Cleaning (Sample):")
print(df[['Job Title', 'Key Skills', 'Functional Area']].head(3).to_string())

for col in ['Job Title', 'Key Skills', 'Functional Area', 'Industry', 'Role Category']:
    df[col] = df[col].apply(clean_text)

print("\nAfter Cleaning (Sample):")
print(df[['Job Title', 'Key Skills', 'Functional Area']].head(3).to_string())

# ==============================================
# STEP 5: FILTER IRRELEVANT JOB TITLES
# ==============================================


print("\n" + "=" * 55)
print("  STEP 5: FILTERING IRRELEVANT JOB TITLES")
print("=" * 55)

remove_words = ["walkin", "walk in", "job description",
                "opening", "vacancy", "urgent", "hiring"]

rows_before = df.shape[0]

for word in remove_words:
    df = df[~df['Job Title'].str.contains(word, case=False)]

rows_after = df.shape[0]

print(f"\nRows Before Filter : {rows_before}")
print(f"Rows After  Filter : {rows_after}")
print(f"Rows Removed       : {rows_before - rows_after}")
print(f"\nRemoved noisy job titles containing words:")
print(f"{remove_words}")

# ==============================================
# STEP 6: USE FUNCTIONAL AREA AS TARGET LABEL
# ==============================================

print("\n" + "=" * 55)
print("  STEP 6: PREPARING TARGET — FUNCTIONAL AREA")
print("=" * 55)

print(f"\nTotal Unique Functional Areas Before Filter : {df['Functional Area'].nunique()}")

# Keep only categories with at least 30 samples
rows_before = df.shape[0]
df = df.groupby('Functional Area').filter(lambda x: len(x) >= 30)
rows_after  = df.shape[0]

print(f"Total Unique Functional Areas After Filter  : {df['Functional Area'].nunique()}")
print(f"\nRows Before : {rows_before}")
print(f"Rows After  : {rows_after}")
print(f"Rows Removed: {rows_before - rows_after}")

print(f"\nFunctional Area Categories and Job Counts:")
print(f"{'No.':<5} {'Category':<45} {'Count':>6}")
print("-" * 58)
counts = df['Functional Area'].value_counts()
for i, (cat, count) in enumerate(counts.items(), 1):
    print(f"{i:<5} {cat:<45} {count:>6}")

# ==============================================
# STEP 7: COMBINE FEATURES FOR MODEL INPUT
# ==============================================

print("\n" + "=" * 55)
print("  STEP 7: COMBINING INPUT FEATURES")
print("=" * 55)

df['combined'] = (
    df['Job Title']  + " " +
    df['Key Skills'] + " " +
    df['Industry']
)

print("\nSample Combined Feature (Row 0):")
print(df['combined'].iloc[0])
print("\nDataset Shape After Combination:", df.shape)

# ==============================================
# STEP 8: LABEL ENCODING
# ==============================================
# LabelEncoder converts text category names into
# numbers so the ML model can process them.
# Example: "IT Software" → 5, "Data Science" → 2
# ==============================================

print("\n" + "=" * 55)
print("  STEP 8: LABEL ENCODING — FUNCTIONAL AREA")
print("=" * 55)

le = LabelEncoder()
df['label'] = le.fit_transform(df['Functional Area'])

print(f"\nTotal Classes (Functional Areas) : {len(le.classes_)}")
print(f"Encoding : Text Category → Numeric Label")
print(f"\nSample Encoded Classes (First 5):")
print(f"{'Label':<8} {'Functional Area'}")
print("-" * 40)
for i in range(min(5, len(le.classes_))):
    print(f"  {i:<6} →  {le.classes_[i]}")
print(f"  ...    →  ...")
print(f"  {len(le.classes_)-1:<6} →  {le.classes_[-1]}")

# ==============================================
# STEP 9: TF-IDF VECTORIZATION
# ==============================================

print("\n" + "=" * 55)
print("  STEP 9: TF-IDF FEATURE EXTRACTION")
print("=" * 55)

tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))

X = tfidf.fit_transform(df['combined'])
y = df['label']

print(f"\nTF-IDF Matrix Shape  : {X.shape}")
print(f"  → {X.shape[0]} job rows")
print(f"  → {X.shape[1]} text features (words/phrases)")

# ==============================================
# STEP 10: TRAIN-TEST SPLIT
# ==============================================

print("\n" + "=" * 55)
print("  STEP 10: TRAIN-TEST SPLIT (80% / 20%)")
print("=" * 55)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining Set : {X_train.shape[0]} samples")
print(f"Testing  Set : {X_test.shape[0]} samples")
print(f"Features     : {X_train.shape[1]}")

# ==============================================
# STEP 11: TRAIN LinearSVC MODEL
# ==============================================

print("\n" + "=" * 55)
print("  STEP 11: TRAINING LinearSVC MODEL")
print("=" * 55)

model = LinearSVC(C=1.0, max_iter=3000)

print("\nAlgorithm   : LinearSVC (Support Vector Classifier)")
print("C value     : 1.0  (regularization parameter)")
print("Max Iter    : 3000")
print("\nTraining... please wait...")

model.fit(X_train, y_train)

print("Training Complete ✅")

# ==============================================
# STEP 12: MODEL EVALUATION
# ==============================================

print("\n" + "=" * 55)
print("  STEP 12: MODEL EVALUATION RESULTS")
print("=" * 55)

y_pred = model.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\n  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")

print(f"""
  What these mean:
  - Accuracy  : {accuracy*100:.1f}% of test predictions were correct
  - Precision : Model is {precision*100:.1f}% precise when it predicts a category
  - Recall    : Model correctly finds {recall*100:.1f}% of actual category instances
  - F1 Score  : Overall balance of precision and recall = {f1:.4f}
""")

# ==============================================
# STEP 13: SAVE MODELS
# ==============================================

print("=" * 55)
print("  STEP 13: SAVING ALL MODELS")
print("=" * 55)

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/svc_model.pkl")
joblib.dump(tfidf, "model/tfidf.pkl")
joblib.dump(le,    "model/label_encoder.pkl")
df.to_csv("model/jobs_cleaned.csv", index=False)

print("\n  model/svc_model.pkl     — Trained LinearSVC model")
print("  model/tfidf.pkl         — TF-IDF vectorizer")
print("  model/label_encoder.pkl — Label encoder for Functional Area")
print("  model/jobs_cleaned.csv  — Cleaned dataset")
print("\n  ✅ All models saved successfully!")
print("  ▶  Now run: streamlit run app.py")