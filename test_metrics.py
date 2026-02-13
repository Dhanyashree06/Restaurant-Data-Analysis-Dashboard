import pandas as pd
import unittest
from app import get_analysis_metrics, clean_words


class TestDashboardMetrics(unittest.TestCase):
    def setUp(self):
        # Create a tiny mock dataset
        self.df = pd.DataFrame({
            'Restaurant Name': ['Rest A', 'Rest B', 'Rest C'],
            'City': ['New York', 'London', 'New York'],
            'Cuisines': ['Italian', 'Italian, Pizza', 'Indian'],
            'Aggregate rating': [4.5, 3.2, 4.8],
            'Rating color': ['Dark Green', 'Yellow', 'Dark Green'],
            'Rating text': ['Excellent service and food', 'Good', 'Excellent wow'],
            'Votes': [100, 50, 200],
            'Price range': [3, 2, 4],
            'Has Online delivery': ['Yes', 'No', 'Yes'],
            'Has Table booking': ['No', 'No', 'Yes']
        })
        self.df['Review Length'] = self.df['Rating text'].apply(
            lambda x: len(clean_words(x)))

    def test_metrics_calculation(self):
        metrics = get_analysis_metrics(self.df)

        self.assertEqual(metrics['top_city'], 'New York')
        self.assertEqual(metrics['best_city'], 'New York')
        self.assertEqual(metrics['highest_votes_rest']
                         ['Restaurant Name'], 'Rest C')
        self.assertIn('Italian', metrics['top_cuisines'])
        self.assertTrue(metrics['review_rating_corr'] != 0)

    def test_empty_df(self):
        metrics = get_analysis_metrics(pd.DataFrame())
        self.assertEqual(metrics['top_city'], 'N/A')


if __name__ == '__main__':
    unittest.main()
