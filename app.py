from collections import Counter
import re
from waitress import serve
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load and Preprocess Data Once
data = pd.read_csv("Dataset.csv")
data['Cuisines'] = data['Cuisines'].fillna("Unknown")
data['Restaurant Name'] = data['Restaurant Name'].fillna("Unknown")
data['City'] = data['City'].fillna("Unknown")
data['Rating text'] = data['Rating text'].fillna("")


def clean_words(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    return re.findall(r'\b\w+\b', text)


# Precalculate Review Length once for all data
data['Review Length'] = data['Rating text'].apply(
    lambda x: len(clean_words(x)))


def get_analysis_metrics(df):
    if df.empty:
        return {
            "top_cuisines": {},
            "top_city": "N/A",
            "best_city": "N/A",
            "price_dist": {},
            "delivery_counts": {},
            "positive_keywords": {},
            "negative_keywords": {},
            "avg_review_length": 0,
            "review_rating_corr": 0,
            "highest_votes_rest": {"Restaurant Name": "N/A", "Votes": 0},
            "lowest_votes_rest": {"Restaurant Name": "N/A", "Votes": 0},
            "votes_rating_corr": 0,
            "price_service": {}
        }

    # 1. Cuisine Analysis
    cuisines_series = df['Cuisines'].str.split(', ').explode()
    top_cuisines = cuisines_series.value_counts().head()

    # 2. City Analysis
    city_count = df['City'].value_counts()
    top_city = city_count.idxmax() if not city_count.empty else "N/A"

    avg_rating_city = df.groupby('City')['Aggregate rating'].mean()
    best_city = avg_rating_city.idxmax() if not avg_rating_city.empty else "N/A"

    # 3. Price Distribution
    price_dist = df['Price range'].value_counts().sort_index()

    # 4. Service Availability
    delivery_counts = df['Has Online delivery'].value_counts()

    # 5. Sentiment Analysis (Keywords)
    positive_words = ["good", "great", "excellent", "amazing", "tasty", "nice"]
    negative_words = ["bad", "poor", "worst", "slow", "dirty", "average"]

    all_words = []
    for review in df['Rating text']:
        all_words.extend(clean_words(review))

    word_count = Counter(all_words)
    positive_keywords = {w: word_count[w]
                         for w in positive_words if word_count[w] > 0}
    negative_keywords = {w: word_count[w]
                         for w in negative_words if word_count[w] > 0}

    # 6. Review Insights
    avg_review_length = df['Review Length'].mean()

    # Correlation needs at least 2 points and non-zero variance
    if len(df) > 1 and df['Review Length'].std() > 0 and df['Aggregate rating'].std() > 0:
        review_rating_corr = df['Review Length'].corr(df['Aggregate rating'])
    else:
        review_rating_corr = 0

    # 7. Voting Trends
    highest_votes_rest = df.loc[df['Votes'].idxmax()]
    lowest_votes_rest = df.loc[df['Votes'].idxmin()]

    if len(df) > 1 and df['Votes'].std() > 0 and df['Aggregate rating'].std() > 0:
        votes_rating_corr = df['Votes'].corr(df['Aggregate rating'])
    else:
        votes_rating_corr = 0

    # 8. Price Range vs Services
    price_service = df.groupby('Price range')[['Has Online delivery', 'Has Table booking']].apply(
        lambda x: (x == 'Yes').mean() * 100
    )

    return {
        "top_cuisines": top_cuisines,
        "top_city": top_city,
        "best_city": best_city,
        "price_dist": price_dist,
        "delivery_counts": delivery_counts,
        "positive_keywords": positive_keywords,
        "negative_keywords": negative_keywords,
        "avg_review_length": round(avg_review_length, 2) if not pd.isna(avg_review_length) else 0,
        "review_rating_corr": round(review_rating_corr, 3) if not pd.isna(review_rating_corr) else 0,
        "highest_votes_rest": highest_votes_rest.to_dict(),
        "lowest_votes_rest": lowest_votes_rest.to_dict(),
        "votes_rating_corr": round(votes_rating_corr, 3) if not pd.isna(votes_rating_corr) else 0,
        "price_service": price_service.to_dict(orient='index')
    }


@app.route("/")
def index():
    search_query = request.args.get('search', '').strip()

    # Filter data based on search
    if search_query:
        mask = (
            data['Restaurant Name'].str.contains(search_query, case=False, na=False, regex=False) |
            data['City'].str.contains(search_query, case=False, na=False, regex=False) |
            data['Cuisines'].str.contains(
                search_query, case=False, na=False, regex=False)
        )
        display_data = data[mask]
        search_results = display_data.head(15).to_dict(orient='records')
    else:
        display_data = data
        search_results = None

    # Calculate metrics for the DISPLAYED (filtered or total) data
    metrics = get_analysis_metrics(display_data)

    return render_template(
        "index.html",
        query=search_query,
        search_results=search_results,
        **metrics
    )


if __name__ == "__main__":
    print("Starting Restaurant Dashboard Server...")
    serve(app, host='0.0.0.0', port=5000)
