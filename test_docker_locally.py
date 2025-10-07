#!/usr/bin/env python3
"""
Local Docker Testing Script
Test the Docker container locally before CI/CD
"""

import requests
import json
import time
import subprocess
import sys

def test_docker_container():
    """Test Docker container functionality"""
    
    print("🧪 Testing Docker Container Locally...")
    
    # Test data
    test_cases = [
        {"text": "This is an amazing product!", "expected": "positive"},
        {"text": "I hate this terrible service", "expected": "negative"},
        {"text": "This is okay I guess", "expected": "either"},
    ]
    
    base_url = "http://localhost:5000"
    
    try:
        # Test health endpoint
        print("\n1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data}")
            
            if not health_data.get('model_loaded') or not health_data.get('vectorizer_loaded'):
                print("⚠️  Warning: Model or vectorizer not loaded properly")
                return False
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
        
        # Test home endpoint
        print("\n2. Testing home endpoint...")
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            print("✅ Home page accessible")
        else:
            print(f"❌ Home page failed: {response.status_code}")
            return False
        
        # Test API prediction endpoint
        print("\n3. Testing API prediction endpoint...")
        for i, test_case in enumerate(test_cases, 1):
            try:
                response = requests.post(
                    f"{base_url}/api/predict", 
                    json={"text": test_case["text"]},
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Test {i}: '{test_case['text'][:30]}...' -> {result.get('sentiment', 'unknown')}")
                else:
                    print(f"❌ Test {i} failed: {response.status_code} - {response.text}")
                    return False
                    
            except Exception as e:
                print(f"❌ Test {i} error: {e}")
                return False
        
        print("\n✅ All Docker tests passed!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Docker container. Is it running on port 5000?")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main test function"""
    print("🐳 Docker Container Local Testing")
    print("="*50)
    
    print("\n📋 Instructions:")
    print("1. Build the Docker image: docker build -t mlops-sentiment-app:test .")
    print("2. Run the container: docker run -d -p 5000:5000 --name test-container mlops-sentiment-app:test")
    print("3. Run this test script: python test_docker_locally.py")
    print("4. Clean up: docker stop test-container && docker rm test-container")
    
    input("\nPress Enter when your Docker container is running on port 5000...")
    
    # Wait a moment for container to fully start
    print("⏳ Waiting for container to initialize...")
    time.sleep(5)
    
    # Run tests
    success = test_docker_container()
    
    if success:
        print("\n🎉 All tests passed! Docker container is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Check the Docker container logs:")
        print("   docker logs test-container")
        sys.exit(1)

if __name__ == "__main__":
    main()