from waitress import serve
from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

data = pd.read_csv("Dataset.csv")
data['Cuisines'] = data['Cuisines'].fillna("Unknown")
data['Restaurant Name'] = data['Restaurant Name'].fillna("Unknown")
data['City'] = data['City'].fillna("Unknown")


@app.route("/")
def index():
    search_query = request.args.get('search', '').strip()

    cuisines_series = data['Cuisines'].str.split(', ').explode()
    top_cuisines = cuisines_series.value_counts().head()

    city_count = data['City'].value_counts()
    top_city = city_count.idxmax()

    avg_rating_city = data.groupby('City')['Aggregate rating'].mean()
    best_city = avg_rating_city.idxmax()

    price_dist = data['Price range'].value_counts().sort_index()

    delivery_counts = data['Has Online delivery'].value_counts()

    search_results = None
    if search_query:
        mask = (
            data['Restaurant Name'].str.contains(search_query, case=False, na=False, regex=False) |
            data['City'].str.contains(search_query, case=False, na=False, regex=False) |
            data['Cuisines'].str.contains(
                search_query, case=False, na=False, regex=False)
        )
        search_results = data[mask].head(15).to_dict(orient='records')

    return render_template(
        "index.html",
        top_cuisines=top_cuisines,
        top_city=top_city,
        best_city=best_city,
        price_dist=price_dist,
        delivery_counts=delivery_counts,
        search_results=search_results,
        query=search_query
    )


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000)
