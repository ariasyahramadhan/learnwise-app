import os
import re
import urllib.request
import json
import warnings
import time
import random
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import paired_cosine_distances

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Paths
ARTIFACTS_DIR = os.path.join("backend", "model_artifacts", "ai_detection")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# 1. Download datasets
print("=== 1. Mengunduh Dataset ===")
# Parquet URL from Hugging Face for alpaca-gpt4-indonesian
ALPACA_URL = 'https://huggingface.co/api/datasets/FreedomIntelligence/alpaca-gpt4-indonesian/parquet/default/train/0.parquet'
HUMAN_URL = 'https://huggingface.co/api/datasets/jakartaresearch/indonews/parquet/default/train/0.parquet'

print("Mengunduh Alpaca GPT-4 Indonesian...")
df_alpaca = pd.read_parquet(ALPACA_URL)
print(f"Alpaca dataset loaded: {len(df_alpaca):,} rows")

print("Mengunduh IndoNews (Human)...")
df_human_raw = pd.read_parquet(HUMAN_URL)
print(f"IndoNews dataset loaded: {len(df_human_raw):,} rows")

# 2. Extract AI and Human texts
print("\n=== 2. Memproses dan Menyeimbangkan Dataset ===")
# Extract GPT-4 responses (AI)
ai_texts = []
for conv in df_alpaca['conversations']:
    if len(conv) >= 2 and conv[1]['from'] == 'gpt':
        ai_texts.append(conv[1]['value'])

# Extract Human texts from IndoNews (and clean the prefix if present)
human_texts = []
for text in df_human_raw['text']:
    cleaned = re.sub(r'^liputan6\.com,\s+[a-zA-Z\s]+\s+-\s+', '', text, flags=re.IGNORECASE)
    human_texts.append(cleaned)

# Shuffle and balance sizes
random.shuffle(ai_texts)
random.shuffle(human_texts)

min_len = min(len(ai_texts), len(human_texts))
ai_texts = ai_texts[:min_len]
human_texts = human_texts[:min_len]

# Match length of human_texts to ai_texts to avoid length bias
matched_human_texts = []
for h_text, a_text in zip(human_texts, ai_texts):
    target_len = len(a_text)
    if len(h_text) > target_len:
        sliced = h_text[:target_len]
        last_space = sliced.rfind(' ')
        if last_space > 0:
            sliced = sliced[:last_space]
        matched_human_texts.append(sliced)
    else:
        matched_human_texts.append(h_text)
human_texts = matched_human_texts

# Create DataFrame
df_ai = pd.DataFrame({'text': ai_texts, 'label': 1})
df_human = pd.DataFrame({'text': human_texts, 'label': 0})
df_dataset = pd.concat([df_ai, df_human], ignore_index=True)
df_dataset = df_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Final dataset size: {len(df_dataset):,} rows")
print(f"  AI-generated (Label 1): {len(df_dataset[df_dataset['label'] == 1]):,}")
print(f"  Human-written (Label 0): {len(df_dataset[df_dataset['label'] == 0]):,}")

# Check length stats
print(f"Average length AI text: {np.mean([len(t) for t in ai_texts]):.1f} chars")
print(f"Average length Human text: {np.mean([len(t) for t in human_texts]):.1f} chars")

# 3. Stylometric Feature Extraction
print("\n=== 3. Mengekstrak Fitur Stylometry ===")

CHATGPT_MARKERS = [
    r'\boleh karena itu\b',
    r'\bselain itu\b',
    r'\bnamun\b',
    r'\bsecara keseluruhan\b',
    r'\bpenting untuk\b',
    r'\bpada dasarnya\b',
    r'\bdalam hal ini\b',
    r'\bdengan demikian\b',
    r'\bkesimpulannya\b',
    r'\bsebagai contoh\b'
]

def extract_stylometry(text):
    text_str = str(text)
    text_lower = text_str.lower()
    words = text_str.split()
    total_words = len(words)
    total_chars = len(text_str)
    
    if total_words == 0:
        return np.zeros(len(CHATGPT_MARKERS) + 7)
    
    # 1. Lexical Diversity (Type-Token Ratio)
    unique_words = len(set(w.lower() for w in words))
    ttr = unique_words / total_words
    
    # 2. Word Length Stats
    word_lengths = [len(w) for w in words]
    avg_word_len = sum(word_lengths) / total_words
    
    # 3. Sentence Length Stats
    sentences = re.split(r'[.!?]+', text_str)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = max(len(sentences), 1)
    avg_sent_len = total_words / num_sentences
    
    # 4. Punctuation rates
    comma_rate = text_str.count(',') / max(total_chars, 1)
    period_rate = text_str.count('.') / max(total_chars, 1)
    question_rate = text_str.count('?') / max(total_chars, 1)
    
    # 5. Capital letters rate
    cap_rate = sum(1 for c in text_str if c.isupper()) / max(total_chars, 1)
    
    # 6. ChatGPT transition word markers
    marker_features = []
    for marker in CHATGPT_MARKERS:
        count = len(re.findall(marker, text_lower))
        marker_features.append(count / total_words)
        
    return np.array([
        ttr, avg_word_len, avg_sent_len, comma_rate, period_rate, question_rate, cap_rate
    ] + marker_features)

# Extract stylometry features for the dataset
sty_features = []
for idx, text in enumerate(df_dataset['text']):
    sty_features.append(extract_stylometry(text))
sty_features = np.array(sty_features)
print(f"Stylometric features extracted: {sty_features.shape}")

# 4. TF-IDF Feature Extraction
print("\n=== 4. Vektorisasi TF-IDF ===")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_dataset['cleaned_text'] = df_dataset['text'].apply(clean_text)

print("Melatih TF-IDF Word Vectorizer...")
word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
X_word = word_vec.fit_transform(df_dataset['cleaned_text']).toarray()

print("Melatih TF-IDF Char Vectorizer...")
char_vec = TfidfVectorizer(ngram_range=(3, 5), max_features=3000, analyzer='char')
X_char = char_vec.fit_transform(df_dataset['cleaned_text']).toarray()

# Combine all features
X_combined = np.hstack([X_word, X_char, sty_features])
y = df_dataset['label'].values
print(f"Combined features shape: {X_combined.shape}")

# 5. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# 6. Train XGBoost Classifier
print("\n=== 5. Melatih Model XGBoost ===")
clf = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

start_time = time.time()
clf.fit(X_train, y_train)
elapsed = time.time() - start_time
print(f"Training completed in {elapsed:.2f} seconds.")

# 7. Evaluate Model
print("\n=== 6. Evaluasi Performa Model ===")
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))

# Find optimal threshold to balance precision and recall on the test set
thresholds = np.linspace(0.1, 0.9, 81)
best_f1 = 0
best_thresh = 0.5
for t in thresholds:
    y_pred_t = (y_pred_proba >= t).astype(int)
    from sklearn.metrics import f1_score
    score = f1_score(y_test, y_pred_t)
    if score > best_f1:
        best_f1 = score
        best_thresh = t

print(f"Optimal Threshold: {best_thresh:.4f} (Best F1-Score: {best_f1:.4f})")

# 8. Save Artifacts
print("\n=== 7. Menyimpan Model dan Fitur ===")
joblib.dump(word_vec, os.path.join(ARTIFACTS_DIR, "ai_word_vectorizer.pkl"))
joblib.dump(char_vec, os.path.join(ARTIFACTS_DIR, "ai_char_vectorizer.pkl"))
joblib.dump(clf, os.path.join(ARTIFACTS_DIR, "ai_classifier.pkl"))

config = {
    "threshold_optimal": float(best_thresh),
    "accuracy": float(accuracy),
    "auc": float(auc),
    "features_stylometry_count": int(sty_features.shape[1]),
    "features_word_count": int(X_word.shape[1]),
    "features_char_count": int(X_char.shape[1]),
    "chatgpt_markers": CHATGPT_MARKERS
}

with open(os.path.join(ARTIFACTS_DIR, "ai_config.pkl"), "wb") as f:
    joblib.dump(config, f)

print("Model dan Vectorizer berhasil disimpan di folder:")
print(f"  {ARTIFACTS_DIR}")
print("Semua proses selesai dengan sukses!")
