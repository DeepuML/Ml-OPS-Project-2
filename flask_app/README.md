# Flask Sentiment Analysis Web Application

This Flask application provides a web interface for the sentiment analysis model trained in the MLOps project.

## Features

- 🎭 **Interactive Web Interface**: User-friendly form for text input and sentiment prediction
- 🔮 **Real-time Analysis**: Instant sentiment classification with confidence scores
- 📊 **Model Performance**: Displays model accuracy and performance metrics
- 🚀 **Production Ready**: Integrates with MLflow model registry
- 🎨 **Responsive Design**: Works on desktop and mobile devices
- 🔒 **Error Handling**: Graceful error handling and user feedback

## Setup Instructions

1. **Install Dependencies**:

   ```bash
   pip install -r flask_requirements.txt
   ```

2. **Environment Configuration**:

   ```bash
   cp .env.example .env
   # Edit .env file with your DagsHub token
   ```

3. **Run the Application**:

   ```bash
   cd flask_app
   python app.py
   ```

4. **Access the Application**:
   - Open your browser and go to `http://localhost:5000`
   - Enter any text and click "Analyze Sentiment"

## API Endpoints

### Web Interface

- `GET /` - Main sentiment analysis page
- `POST /predict` - Form submission for sentiment prediction

### API Endpoints

- `POST /api/predict` - JSON API for sentiment prediction
- `GET /health` - Health check endpoint

## API Usage Example

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

Response:

```json
{
  "prediction": "😊 Positive",
  "prediction_value": 1,
  "confidence": "High",
  "original_text": "I love this product!",
  "cleaned_text": "love product"
}
```

## Model Information

- **Model Type**: Logistic Regression
- **Accuracy**: 79.41%
- **Features**: Bag of Words (TF-IDF)
- **Preprocessing**: NLTK-based text cleaning
- **Registry**: MLflow Model Registry (Production stage)

## File Structure

```
flask_app/
├── app.py              # Main Flask application
├── .env.example        # Environment variables template
├── templates/
│   └── index.html      # Web interface template
├── static/
│   └── style.css       # Custom CSS styles
└── README.md           # This file
```

## Deployment

For production deployment:

1. Set `FLASK_ENV=production` in `.env`
2. Use a production WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

## Troubleshooting

- **Model Loading Issues**: Ensure the model is registered in MLflow and accessible
- **Authentication Errors**: Check your DagsHub token in the `.env` file
- **NLTK Data**: The app will automatically download required NLTK data
- **Port Conflicts**: Change the port in `app.py` if 5000 is already in use

## Performance

- Model loads once at startup for optimal performance
- Caching implemented for frequent predictions
- Responsive design with loading indicators
- Error boundaries for graceful failure handling
