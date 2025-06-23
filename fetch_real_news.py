import requests
import pandas as pd
from datetime import datetime
import os

# === CONFIG ===
API_KEY = '218098e7f2954238973cc354a01980d4'  # Your NewsAPI key
NEWS_API_URL = (
    f'https://newsapi.org/v2/top-headlines?language=en&pageSize=100&apiKey={API_KEY}'
)
TRUE_CSV_PATH = os.path.join('model', 'true.csv')  # Path to existing True.csv

def fetch_today_news():
    response = requests.get(NEWS_API_URL)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch news: {response.status_code}")

    articles = response.json().get('articles', [])
    news_data = []

    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        content = article.get("content", "")

        full_text = f"{title}. {description} {content}".strip()
        if full_text:
            news_data.append({
                "title": title,
                "text": full_text,
                "subject": "real",
                "date": article.get("publishedAt", datetime.utcnow().isoformat())
            })

    return pd.DataFrame(news_data)

def update_true_csv():
    print("🔄 Fetching latest real news articles...")
    new_data = fetch_today_news()

    if os.path.exists(TRUE_CSV_PATH):
        old_data = pd.read_csv(TRUE_CSV_PATH)
        combined_data = pd.concat([old_data, new_data], ignore_index=True)
        combined_data.drop_duplicates(subset=["text"], inplace=True)
    else:
        combined_data = new_data

    combined_data.to_csv(TRUE_CSV_PATH, index=False)
    print(f"✅ Updated {TRUE_CSV_PATH} with {len(new_data)} new articles.")
    print(f"📊 Total real articles now: {len(combined_data)}")

if __name__ == "__main__":
    update_true_csv()
