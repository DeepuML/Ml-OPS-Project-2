import os
os.environ['FLASK_ENV'] = 'testing'
os.environ['CI'] = 'true'

from flask_app.app import app

# Initialize the app to load models
with app.app_context():
    # Get the loaded models from globals
    from flask_app.app import model, vectorizer

print(f"Model type: {type(model)}")
print(f"Model str: {str(type(model))}")
print(f"Contains 'mlflow': {str(type(model)).find('mlflow') != -1}")
print(f"Model module: {getattr(model.__class__, '__module__', 'No module')}")

# Test what will happen in prediction
if str(type(model)).find('mlflow') != -1:
    print("✅ MLflow model detected - will use DataFrame input")
else:
    print("❌ Not detected as MLflow model - will use sparse matrix input")