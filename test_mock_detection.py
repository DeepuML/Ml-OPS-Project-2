import os
import sys
sys.path.append('.')

# Force testing environment
os.environ['FLASK_ENV'] = 'testing'
os.environ['CI'] = 'true'

# Now import Flask app which should create mock models
from flask_app.app import app

# Initialize the app to load models
with app.app_context():
    from flask_app.app import model, vectorizer
    
    print(f"Mock Model type: {type(model)}")
    print(f"Mock Model str: {str(type(model))}")
    print(f"Contains 'mlflow': {str(type(model)).find('mlflow') != -1}")
    print(f"Mock Model module: {getattr(model.__class__, '__module__', 'No module')}")
    
    # Test what will happen in prediction
    if str(type(model)).find('mlflow') != -1:
        print("✅ Mock MLflow model detected - will use DataFrame input")
    else:
        print("❌ Mock model not detected as MLflow - will use sparse matrix input")
        
    # Test if we can call predict method
    print(f"Model has predict method: {hasattr(model, 'predict')}")
    
    # Test predict method with a simple input
    import pandas as pd
    test_df = pd.DataFrame({'0': [1], '1': [0]})
    try:
        result = model.predict(test_df)
        print(f"✅ Model.predict with DataFrame works: {result}")
    except Exception as e:
        print(f"❌ Model.predict with DataFrame failed: {e}")