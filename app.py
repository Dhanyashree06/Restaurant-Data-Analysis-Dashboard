from collections import Counter
import re
from waitress import serve
from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load and preprocess data
try:
    data = pd.read_csv("Dataset.csv")
    data['Cuisines'] = data['Cuisines'].fillna("Unknown")
    data['Restaurant Name'] = data['Restaurant Name'].fillna("Unknown")
    data['City'] = data['City'].fillna("Unknown")
    data['Rating text'] = data['Rating text'].fillna("")
except Exception as e:
    print(f"Error loading data: {e}")
    data = pd.DataFrame()


def clean_words(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    return re.findall(r'\b\w+\b', text)


if not data.empty:
    data['Review Length'] = data['Rating text'].apply(
        lambda x: len(clean_words(x)))
else:
    data['Review Length'] = 0


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

    # Top Cuisines
    cuisines_series = df['Cuisines'].str.split(', ').explode()
    top_cuisines = cuisines_series.value_counts().head(5).to_dict()

    # City Analysis
    city_count = df['City'].value_counts()
    top_city = city_count.idxmax() if not city_count.empty else "N/A"

    avg_rating_city = df.groupby('City')['Aggregate rating'].mean()
    best_city = avg_rating_city.idxmax() if not avg_rating_city.empty else "N/A"

    # Price Distribution
    price_dist = df['Price range'].value_counts().sort_index().to_dict()

    # Online Delivery
    delivery_counts = df['Has Online delivery'].value_counts().to_dict()

    # Sentiment Keywords
    positive_words = ["good", "great", "excellent",
                      "amazing", "tasty", "nice", "fresh", "friendly"]
    negative_words = ["bad", "poor", "worst", "slow",
                      "dirty", "average", "overpriced", "rude"]

    all_words = []
    for review in df['Rating text']:
        all_words.extend(clean_words(review))

    word_count = Counter(all_words)
    positive_keywords = {w: word_count[w]
                         for w in positive_words if word_count[w] > 0}
    negative_keywords = {w: word_count[w]
                         for w in negative_words if word_count[w] > 0}

    # Correlations and averages
    avg_review_length = float(df['Review Length'].mean())

    if len(df) > 1 and df['Review Length'].std() > 0 and df['Aggregate rating'].std() > 0:
        review_rating_corr = float(
            df['Review Length'].corr(df['Aggregate rating']))
    else:
        review_rating_corr = 0.0

    # Votes
    try:
        highest_votes_idx = df['Votes'].idxmax()
        highest_votes_rest = df.loc[highest_votes_idx].to_dict()
    except:
        highest_votes_rest = {"Restaurant Name": "N/A", "Votes": 0}

    if len(df) > 1 and df['Votes'].std() > 0 and df['Aggregate rating'].std() > 0:
        votes_rating_corr = float(df['Votes'].corr(df['Aggregate rating']))
    else:
        votes_rating_corr = 0.0

    # Service Availability by Price Range
    price_service_df = df.groupby('Price range')[['Has Online delivery', 'Has Table booking']].apply(
        lambda x: (x == 'Yes').mean() * 100
    )
    price_service = price_service_df.to_dict(orient='index')

    return {
        "top_cuisines": top_cuisines,
        "top_city": top_city,
        "best_city": best_city,
        "price_dist": price_dist,
        "delivery_counts": delivery_counts,
        "positive_keywords": positive_keywords,
        "negative_keywords": negative_keywords,
        "avg_review_length": round(avg_review_length, 2),
        "review_rating_corr": round(review_rating_corr, 3),
        "highest_votes_rest": highest_votes_rest,
        "votes_rating_corr": round(votes_rating_corr, 3),
        "price_service": price_service
    }


@app.route("/")
def index():
    search_query = request.args.get('search', '').strip()

    if not data.empty:
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
    else:
        display_data = pd.DataFrame()
        search_results = None

    metrics = get_analysis_metrics(display_data)

    return render_template(
        "index.html",
        query=search_query,
        search_results=search_results,
        **metrics
    )


if __name__ == "__main__":
    print("Starting Restaurant Dashboard Server on http://localhost:5000")
    serve(app, host='0.0.0.0', port=5000)
