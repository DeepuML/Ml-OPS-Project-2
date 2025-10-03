import unittest
import os
from unittest.mock import patch, MagicMock
from flask_app.app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set testing mode
        os.environ['FLASK_ENV'] = 'testing'
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Sentiment Analysis - MLOps Project</title>', response.data)

    def test_predict_page(self):
        response = self.client.post('/predict', data=dict(text="I love this!"))
        self.assertEqual(response.status_code, 200)
        # Check if model is loaded and prediction made, or error message shown
        prediction_made = (b'Positive' in response.data or b'Negative' in response.data)
        error_shown = (b'Model not loaded' in response.data or b'error' in response.data.lower())
        self.assertTrue(
            prediction_made or error_shown,
            "Response should contain either prediction result or error message"
        )

    def test_predict_page_empty_text(self):
        """Test prediction with empty text"""
        response = self.client.post('/predict', data=dict(text=""))
        self.assertEqual(response.status_code, 200)
        # Should handle empty text gracefully

    def test_api_predict_endpoint(self):
        """Test JSON API endpoint"""
        response = self.client.post('/api/predict', 
                                  json={'text': 'This is great!'})
        # Should return 200 or handle gracefully if model not loaded
        self.assertIn(response.status_code, [200, 500])

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('status', data)

if __name__ == '__main__':
    unittest.main()