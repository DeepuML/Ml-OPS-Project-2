#!/usr/bin/env python3
"""
Test script to verify model loading in different environments
"""

import os
import sys

# Add the flask_app directory to the path
sys.path.append('flask_app')

def test_model_loading():
    """Test model loading in different environments"""
    
    print("🔍 Testing Model Loading...")
    
    # Test 1: Normal environment (should try to load real model)
    print("\n1️⃣ Testing Normal Environment:")
    os.environ.pop('FLASK_ENV', None)  # Remove if exists
    os.environ.pop('CI', None)         # Remove if exists
    
    try:
        from flask_app.app import model, vectorizer
        if model is not None and vectorizer is not None:
            print("   ✅ Model and vectorizer loaded successfully")
        else:
            print("   ⚠️  Model or vectorizer is None (expected if no local files)")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
    
    # Test 2: Testing environment (should create mock model)
    print("\n2️⃣ Testing CI/Testing Environment:")
    os.environ['FLASK_ENV'] = 'testing'
    os.environ['CI'] = 'true'
    
    # Clear the module cache to force re-import
    modules_to_remove = [k for k in sys.modules.keys() if k.startswith('flask_app')]
    for module in modules_to_remove:
        del sys.modules[module]
    
    try:
        from flask_app.app import model, vectorizer
        if model is not None and vectorizer is not None:
            print("   ✅ Mock model and vectorizer created successfully")
            
            # Test mock prediction
            try:
                # Create dummy data for testing
                import pandas as pd
                dummy_data = pd.DataFrame([[0.5, 0.3, 0.2]], columns=['0', '1', '2'])
                prediction = model.predict(dummy_data)
                print(f"   ✅ Mock prediction test: {prediction}")
            except Exception as e:
                print(f"   ⚠️  Mock prediction failed: {e}")
        else:
            print("   ❌ Mock model creation failed")
    except Exception as e:
        print(f"   ❌ Error in testing environment: {e}")
    
    print("\n🎯 Model loading test completed!")

if __name__ == "__main__":
    test_model_loading()