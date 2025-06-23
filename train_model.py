import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# === 📁 Path Definitions ===
true_path = "dataset/True.csv"
fake_path = "dataset/Fake.csv"
real_news_today = "dataset/RealNews_20250617.csv"  # Optional new data
model_dir = "model"
model_path = os.path.join(model_dir, "model.pkl")
vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")

# === 📥 Load Core Datasets ===
true_df = pd.read_csv(true_path)
fake_df = pd.read_csv(fake_path)
true_df["label"] = 1  # Real
fake_df["label"] = 0  # Fake

# === 📥 Optional: Load Dynamic Real News ===
if os.path.exists(real_news_today):
    real_df = pd.read_csv(real_news_today)
    real_df["label"] = 1
    print(f"✅ Loaded {len(real_df)} new real news articles.")
else:
    real_df = pd.DataFrame(columns=["text", "label"])
    print("⚠️ No new real news found. Proceeding with base dataset only.")

# === 🔗 Merge and Shuffle ===
data = pd.concat([
    true_df[["text", "label"]],
    fake_df[["text", "label"]],
    real_df[["text", "label"]]
], ignore_index=True)

data.dropna(subset=["text"], inplace=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# === ✂️ Split ===
x_train, x_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.25, random_state=42
)

# === 🔤 Vectorize ===
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# === 🧠 Train Model ===
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# === 📈 Evaluation ===
y_pred = pac.predict(tfidf_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# === 💾 Save Model and Vectorizer ===
os.makedirs(model_dir, exist_ok=True)

with open(model_path, "wb") as f:
    pickle.dump(pac, f)

with open(vectorizer_path, "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

print("✅ Model and vectorizer saved to 'model/' folder.")
